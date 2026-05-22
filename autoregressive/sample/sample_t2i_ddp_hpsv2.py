import os
import sys
import json
import torch
import argparse
import math
from tqdm import tqdm
from PIL import Image
import torch.distributed as dist

# Import LlamaGen dependencies
from autoregressive.models.gpt import GPT_models
from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.sample.sample_t2i_ddp import generate

def tensor2pil(image: torch.Tensor):
    assert image.ndim == 3
    return Image.fromarray(((image + 1.0) * 127.5).clamp(0.0, 255.0).to(torch.uint8).permute(1, 2, 0).detach().cpu().numpy())

def load_local_hpdv2_prompts(hpdv2_path):
    styles = ['anime', 'concept-art', 'paintings', 'photo']
    all_prompts = {}
    
    for style in styles:
        # First try benchmark_style.json
        json_path = os.path.join(hpdv2_path, "benchmark", f"benchmark_{style}.json")
        if not os.path.exists(json_path):
            # If it is missing, fall back to style.json
            json_path = os.path.join(hpdv2_path, "benchmark", f"{style}.json")
            
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Cannot find HPDv2 prompt file: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        prompts = []
        for item in data:
            # Support multiple JSON layouts (dict lists or plain string lists)
            if isinstance(item, dict):
                prompts.append(item.get("prompt", ""))
            else:
                prompts.append(item)
        all_prompts[style] = prompts
        
    return all_prompts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--t5-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--hpdv2-path", type=str, required=True, help="Local HPDv2 dataset path") # Added for the public release
    
    parser.add_argument("--gpt-model", type=str, default="GPT-XL")
    parser.add_argument("--vq-model", type=str, default="VQ-16")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--precision", type=str, default="bf16")
    args = parser.parse_args()

    # Initialize DDP (correct order: set_device first, then init_process_group)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    # 1. Parse local HPDv2 prompts
    all_prompts = load_local_hpdv2_prompts(args.hpdv2_path)
    flat_prompts = []
    
    for style, prompts in all_prompts.items():
        # Pre-create the directory for each style
        if rank == 0:
            os.makedirs(os.path.join(args.save_dir, style), exist_ok=True)
        for idx, prompt in enumerate(prompts):
            flat_prompts.append((style, idx, prompt))
    
    dist.barrier() # Wait for the main process to finish creating directories

    # Assign work to the current GPU
    local_prompts = flat_prompts[rank::world_size]

    # 2. Load models
    if rank == 0: print(f"Loading Models on rank 0...")
    precision_type = torch.bfloat16 if args.precision == "bf16" else torch.float32
    
    # T5
    t5_model = T5Embedder(device=device, local_cache=True, cache_dir=args.t5_path, dir_or_name="flan-t5-xl", torch_dtype=precision_type)
    
    # VQ
    vq_model = VQ_models[args.vq_model](codebook_size=16384, codebook_embed_dim=8)
    vq_model.to(device).eval()
    vq_ckpt = torch.load(args.vq_ckpt, map_location='cpu')
    vq_model.load_state_dict(vq_ckpt['model'] if 'model' in vq_ckpt else vq_ckpt)

    # GPT
    latent_size = args.image_size // 16
    gpt_model = GPT_models[args.gpt_model](block_size=latent_size**2, cls_token_num=120, model_type='t2i')
    gpt_model.to(device).eval()
    gpt_ckpt = torch.load(args.gpt_ckpt, map_location='cpu')
    state_dict = gpt_ckpt['model'] if 'model' in gpt_ckpt else (gpt_ckpt['state_dict'] if 'state_dict' in gpt_ckpt else gpt_ckpt)
    gpt_model.load_state_dict({k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}, strict=False)

    # 3. Batched generation
    if rank == 0: print(f"Start HPSv2 Sampling. Total prompts: {len(flat_prompts)}, per GPU: {len(local_prompts)}")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(local_prompts), args.batch_size), disable=(rank != 0)):
            batch = local_prompts[i : i + args.batch_size]
            b_styles = [x[0] for x in batch]
            b_idxs = [x[1] for x in batch]
            b_texts = [x[2] for x in batch]

            c_indices, c_masks = t5_model.get_text_embeddings(b_texts)
            # [Fix] Apply the text mask to embeddings so padding noise does not corrupt generation
            c_indices = c_indices * c_masks[:, :, None]
            
            with torch.autocast(device_type='cuda', dtype=precision_type):
                # Call the autoregressive generate function
                with torch.no_grad():
                    index_sample = generate(
                        gpt_model, c_indices, latent_size**2, 
                        c_masks, 
                        cfg_scale=args.cfg_scale, 
                        temperature=1.0, 
                        top_k=1000, 
                        top_p=1.0,
                        sample_logits=True
                    )
                samples = vq_model.decode_code(index_sample, [len(b_texts), 8, latent_size, latent_size])
                samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save images using the HPSv2 format: <style>/0000X.jpg
            for j in range(len(samples)):
                img = Image.fromarray(samples[j])
                save_path = os.path.join(args.save_dir, b_styles[j], f"{b_idxs[j]:05d}.jpg")
                img.save(save_path, format="JPEG", quality=100)

    dist.barrier()
    if rank == 0: print(">>> HPSv2 Sampling Complete!")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()