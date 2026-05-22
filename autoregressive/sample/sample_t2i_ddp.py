# autoregressive/sample/v0_sample.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import os
import math
import json
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Ensure these modules are available in PYTHONPATH
from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate

def load_gpt_checkpoint(model, ckpt_path):
    """
    Compatible with both legacy checkpoint files that store the raw state_dict and the common {'model': ...} format.
    """
    print(f"Loading GPT from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # 1. Resolve the state_dict payload
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "module" in checkpoint:
        state_dict = checkpoint["module"]
    else:
        # Legacy case: the checkpoint itself is already a state_dict
        state_dict = checkpoint

    # 2. Strip possible key prefixes (for example _orig_mod or module.)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '').replace('_orig_mod.', '')
        new_state_dict[new_k] = v
        
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"GPT Loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

def main(args):
    # --- 1. Init DDP ---
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    
    # Set random seed
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    
    if rank == 0:
        print(f"Starting Sampling. World Size: {world_size}, Checkpoint: {args.gpt_ckpt}")

    # --- 2. Load Models ---
    # A. VQ Model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim
    )
    vq_model.to(device).eval()
    vq_ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(vq_ckpt['model'] if 'model' in vq_ckpt else vq_ckpt)
    
    # B. T5 Model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )

    # C. GPT Model
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    
    load_gpt_checkpoint(gpt_model, args.gpt_ckpt)
    gpt_model.eval()

    if args.compile:
        gpt_model = torch.compile(gpt_model)

    # --- 3. Prepare Prompt List (Optimized) ---    
    # Candidate prompt column names, matched case-insensitively in priority order
    target_cols = ['caption', 'prompt', 'text', 'prompts', 'captions']
    
    try:
        # 1. First try standard structured parsing (CSV/TSV)
        if args.prompt_csv.endswith('.tsv'):
            df = pd.read_csv(args.prompt_csv, sep='\t')
        else:
            # For .csv, try standard parsing first. Content with commas will fall back through the exception path if needed.
            df = pd.read_csv(args.prompt_csv)

        # 2. Find the target text column
        # Match after lowercasing DataFrame column names
        found_col = next((c for c in df.columns if c.lower() in target_cols), None)
        
        if found_col:
            print(f"Load prompts from column: '{found_col}'")
            prompt_list = df[found_col].astype(str).tolist()
        else:
            # If no header matches, fall back to the first column
            print("No header matched, using the first column as prompts.")
            prompt_list = df.iloc[:, 0].astype(str).tolist()

    except Exception as e:
        # 3. Core fallback logic
        # If pd.read_csv fails (for example because commas break the CSV parsing)
        # or if the file is just a plain text list rather than a true CSV
        # fall back to the most robust line-by-line reader
        print(f"Standard CSV parsing failed ({str(e)}). Falling back to line-by-line reading.")
        
        with open(args.prompt_csv, 'r', encoding='utf-8') as f:
            # Strip whitespace and drop empty lines
            prompt_list = [line.strip() for line in f if line.strip()]

    print(f"Total prompts loaded: {len(prompt_list)}")
    
    # Validate prompt availability before batching
    if len(prompt_list) == 0:
        raise ValueError("Prompt list is empty!")

    # --- 4. Distribute Work ---
    # Compute the requested total number of samples
    num_fid_samples = args.num_fid_samples if args.num_fid_samples > 0 else len(prompt_list)
    # Truncate or repeat prompts to match num_fid_samples
    if len(prompt_list) < num_fid_samples:
        prompt_list = (prompt_list * (num_fid_samples // len(prompt_list) + 1))[:num_fid_samples]
    else:
        prompt_list = prompt_list[:num_fid_samples]
    
    # ================= [Patch start] =================
    # Let rank 0 write captions.txt
    if rank == 0:
        # [Critical fix] Create the directory before opening captions.txt
        os.makedirs(args.sample_dir, exist_ok=True) 
        
        captions_path = os.path.join(args.sample_dir, "captions.txt")
        print(f"Saving captions to {captions_path} ...")
        with open(captions_path, "w", encoding="utf-8") as f:
            for p in prompt_list:
                f.write(p.strip() + "\n")
    # ================= [Patch end] =================

    # Assign the local prompt shard for this GPU
    local_prompts = prompt_list[rank::world_size]
    
    # --- 5. Output Directory Setup ---
    # Ensure the output path exists before each rank writes
    if rank == 0:
        os.makedirs(os.path.join(args.sample_dir, "images"), exist_ok=True)
    dist.barrier()

    # --- 6. Generation Loop ---
    # Process prompts in batches
    batch_size = args.per_proc_batch_size
    pbar = tqdm(range(0, len(local_prompts), batch_size), disable=(rank != 0))
    
    total_generated = 0
    for i in pbar:
        batch_prompts = local_prompts[i : i + batch_size]
        
        # T5 Embedding
        caption_embs, emb_masks = t5_model.get_text_embeddings(batch_prompts)
        
        # Input Formatting (Left/Right Padding handling)
        # Keep the reference padding behavior here. T5 outputs are usually right-padded, so sampling follows the training-time convention.
        # Here we directly use caption_embs
        c_indices = caption_embs * emb_masks[:,:, None]
        
        # Generate Tokens
        with torch.no_grad():
            index_sample = generate(
                gpt_model, c_indices, latent_size ** 2, 
                emb_masks,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature, 
                top_k=args.top_k,
                top_p=args.top_p, 
                sample_logits=True, 
            )
        
        # VQ Decode
        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
        with torch.no_grad():
            samples = vq_model.decode_code(index_sample, qzshape)
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            
        # Save Images
        for j, img_arr in enumerate(samples):
            # Globally unique image index
            global_idx = (i + j) * world_size + rank
            img = Image.fromarray(img_arr)
            save_path = os.path.join(args.sample_dir, "images", f"{global_idx:06d}.png")
            img.save(save_path)
    
    dist.barrier()
    if rank == 0:
        print(f"Sampling finished. Saved to {args.sample_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--t5-path", type=str, required=True)
    parser.add_argument("--prompt-csv", type=str, required=True)
    parser.add_argument("--sample-dir", type=str, required=True)
    
    # Model Config (Should match training)
    parser.add_argument("--gpt-model", type=str, default="GPT-XL")
    parser.add_argument("--gpt-type", type=str, default="t2i")
    parser.add_argument("--vq-model", type=str, default="VQ-16")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--cls-token-num", type=int, default=120)
    parser.add_argument("--t5-model-type", type=str, default="flan-t5-xl")
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    
    # Sampling Config
    parser.add_argument("--precision", type=str, default='bf16')
    parser.add_argument("--compile", action='store_true')
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    # Distributed Config
    parser.add_argument("--per-proc-batch-size", type=int, default=16)
    parser.add_argument("--num-fid-samples", type=int, default=30000)
    parser.add_argument("--global-seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)