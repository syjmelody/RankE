import torch
import torch.distributed as dist
import os
import math
import json
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
import sys

# Faster initialization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

# LlamaGen Imports
try:
    from tokenizer.tokenizer_image.vq_model import VQ_models
    from language.t5 import T5Embedder
    from autoregressive.models.gpt import GPT_models
    from autoregressive.models.generate import generate
except ImportError:
    sys.path.append(os.getcwd())
    from tokenizer.tokenizer_image.vq_model import VQ_models
    from language.t5 import T5Embedder
    from autoregressive.models.gpt import GPT_models
    from autoregressive.models.generate import generate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_checkpoint_robust(model, ckpt_path, model_name="model"):
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"[{model_name}] Loading checkpoint from {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint.get("ema", checkpoint)))
    else:
        state_dict = checkpoint
    
    new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

def main(args):
    # --- 1. Init DDP ---
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)

    # --- 2. Load Models ---
    # A. VQ Model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim
    )
    vq_model.to(device).eval()
    load_checkpoint_robust(vq_model, args.vq_ckpt, "VQ-VAE")

    # B. GPT Model
    precision_map = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
    dtype = precision_map[args.precision]
    latent_size = args.image_size // args.downsample_size
    
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device)
    load_checkpoint_robust(gpt_model, args.gpt_ckpt, "GPT")
    gpt_model.to(dtype=dtype).eval()
    
    if args.compile:
        gpt_model = torch.compile(gpt_model)

    # C. T5 Model
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=dtype,
        model_max_length=args.t5_feature_max_len,
    )

    dist.barrier()

    # --- 3. Prepare GenEval Data ---
    if rank == 0:
        print(f"Loading GenEval prompts from: {args.prompts}")
    
    with open(args.prompts, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f.readlines() if line.strip()]

    # Distribute work across DDP ranks
    local_indices = list(range(rank, len(all_lines), world_size))
    local_lines = [all_lines[i] for i in local_indices]

    # --- Compute batching parameters ---
    # Determine how many prompts each forward pass should cover
    # args.batch_size: total images per GPU (for example 32)
    # args.repeat: images generated per prompt (GenEval default: 4)
    prompts_per_batch = max(1, args.batch_size // args.repeat)
    
    if rank == 0:
        print(f"Total prompts: {len(all_lines)}")
        print(f"Total Batch Size per GPU: {args.batch_size}")
        print(f"Prompts per Batch: {prompts_per_batch} (generating {prompts_per_batch * args.repeat} images per step)")
        print(f"Sampling Target Directory: {args.save_dir}")

    # --- 4. Generation Loop (Batched) ---
    # Compute the total number of tqdm steps
    total_steps = (len(local_lines) + prompts_per_batch - 1) // prompts_per_batch
    
    for i in tqdm(range(0, len(local_lines), prompts_per_batch), disable=(rank != 0), total=total_steps):
        # A. Gather prompts for the current batch
        batch_lines = local_lines[i : i + prompts_per_batch]
        batch_global_indices = local_indices[i : i + prompts_per_batch]
        
        flat_prompts = []
        meta_infos = [] # Used during saving to recover (global_idx, sub_idx, line_str)
        
        for line, g_idx in zip(batch_lines, batch_global_indices):
            metadata = json.loads(line)
            p_text = metadata['prompt']
            
            # Repeat each prompt args.repeat times and flatten into flat_prompts
            flat_prompts.extend([p_text] * args.repeat)
            
            # Record per-image metadata
            for r in range(args.repeat):
                meta_infos.append((g_idx, r, line))
        
        # B. Inference
        # T5 Encode
        caption_embs, emb_masks = t5_model.get_text_embeddings(flat_prompts)
        
        # Padding
        if not args.no_left_padding:
            new_emb_masks = torch.flip(emb_masks, dims=[-1])
            new_caption_embs = []
            for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
                valid_num = int(emb_mask.sum().item())
                new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                new_caption_embs.append(new_caption_emb)
            new_caption_embs = torch.stack(new_caption_embs)
        else:
            new_caption_embs, new_emb_masks = caption_embs, emb_masks

        c_indices = new_caption_embs * new_emb_masks[:,:, None]
        c_emb_masks = new_emb_masks

        # GPT Generate
        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
        index_sample = generate(
            gpt_model, c_indices, latent_size ** 2, 
            c_emb_masks,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature, 
            top_k=args.top_k,
            top_p=args.top_p, 
            sample_logits=True, 
        )
        
        # VQ Decode
        samples = vq_model.decode_code(index_sample, qzshape) 
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # C. Save Results
        for img_idx, img_arr in enumerate(samples):
            # Recover the prompt metadata for this image
            g_idx, sub_idx, line_str = meta_infos[img_idx]
            
            prompt_dir = os.path.join(args.save_dir, str(g_idx).zfill(5))
            sample_dir = os.path.join(prompt_dir, "samples")
            
            # Only create the directory and write metadata for the first image of each prompt
            # This avoids concurrent writes and redundant I/O across repeated samples
            if sub_idx == 0:
                os.makedirs(sample_dir, exist_ok=True)
                meta_path = os.path.join(prompt_dir, "metadata.jsonl")
                # Simple guard to avoid duplicate writes
                if not os.path.exists(meta_path):
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        f.write(line_str + "\n")

            # Save the image
            img = Image.fromarray(img_arr)
            save_path = os.path.join(sample_dir, f"{str(sub_idx).zfill(5)}.png")
            img.save(save_path)

    dist.barrier()
    if rank == 0:
        print("GenEval Sampling Finished.")
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, required=True, help="GenEval evaluation_metadata.jsonl")
    parser.add_argument("--save-dir", type=str, required=True)
    
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    
    parser.add_argument("--gpt-model", type=str, default="GPT-XL")
    parser.add_argument("--gpt-type", type=str, default="t2i")
    parser.add_argument("--vq-model", type=str, default="VQ-16")
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--cls-token-num", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--no-left-padding", action='store_true')
    
    parser.add_argument("--precision", type=str, default='bf16')
    parser.add_argument("--compile", action='store_true')
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--repeat", type=int, default=4, help="Samples per prompt")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    
    # Batch size parameter
    parser.add_argument("--batch-size", type=int, default=32, help="Total batch size per GPU (images)")

    args = parser.parse_args()
    main(args)
