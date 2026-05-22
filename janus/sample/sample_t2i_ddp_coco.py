import os
import sys
import argparse
import json
import pandas as pd
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from tqdm import tqdm
import math

from transformers import DynamicCache

# Import Janus modules directly (relying on PYTHONPATH set by the shell script)
from janus.models import MultiModalityCausalLM, VLChatProcessor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def setup_ddp(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda")
        
    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return rank, world_size, device


def load_stitched_janus_model(base_path, target_ckpt_path, device):
    print(f">>> Loading Base Model from {base_path}")
    processor = VLChatProcessor.from_pretrained(base_path)
    
    # Protect against Meta Tensor initialization failures
    original_linspace = torch.linspace
    def patched_linspace(*a, **k):
        k['device'] = 'cpu'
        return original_linspace(*a, **k)
    torch.linspace = patched_linspace
    
    model = MultiModalityCausalLM.from_pretrained(
        base_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        _fast_init=False
    )
    torch.linspace = original_linspace
    
    if base_path != target_ckpt_path:
        print(f">>> Stitching SFT Weights from {target_ckpt_path}")
        import os
        state_dict = None
        if os.path.exists(os.path.join(target_ckpt_path, "model.safetensors")):
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(target_ckpt_path, "model.safetensors"))
        elif os.path.exists(os.path.join(target_ckpt_path, "pytorch_model.bin")):
            state_dict = torch.load(os.path.join(target_ckpt_path, "pytorch_model.bin"), map_location='cpu')
        else:
            raise FileNotFoundError(f"Cannot find valid weights (safetensors/bin) in {target_ckpt_path}")
            
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f">>> SFT Weights Overwritten. Missing keys: {len(missing)}")
    
    model = model.to(device).eval()
    return model, processor


@torch.no_grad()
def janus_generate(
    model: MultiModalityCausalLM,
    processor: VLChatProcessor,
    prompts: list[str],
    cfg_scale: float = 5.0,
    temperature: float = 1.0,
    top_k: int = 50,
    image_token_num: int = 576,
    img_size: int = 384,
    patch_size: int = 16
):
    bs = len(prompts)
    input_ids_list = []
    
    for prompt in prompts:
        conversation = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt="",
        )
        full_prompt = sft_format + processor.image_start_tag
        input_ids_list.append(processor.tokenizer.encode(full_prompt))

    # =====================================================================
    # [Core fix 1] Strict left padding and attention-mask construction
    # =====================================================================
    max_len = max(len(ids) for ids in input_ids_list)
    input_ids_tensor = torch.full((bs, max_len), processor.pad_id, dtype=torch.long, device=model.device)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long, device=model.device)
    
    for i, ids in enumerate(input_ids_list):
        input_ids_tensor[i, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=model.device)
        attention_mask[i, -len(ids):] = 1

    # =====================================================================
    # [Core fix 2] Build CFG unconditional inputs safely without corrupting <BOS>
    # =====================================================================
    use_cfg = cfg_scale > 1.0
    bs_expanded = bs * 2 if use_cfg else bs
    
    if use_cfg:
        tokens = torch.zeros((bs_expanded, max_len), dtype=torch.long, device=model.device)
        batch_mask = torch.zeros((bs_expanded, max_len), dtype=torch.long, device=model.device)
        
        # Even rows: conditional branch
        tokens[::2] = input_ids_tensor
        batch_mask[::2] = attention_mask
        
        # Odd rows: unconditional branch
        tokens[1::2] = input_ids_tensor.clone()
        batch_mask[1::2] = attention_mask.clone()
        
        for i, ids in enumerate(input_ids_list):
            valid_len = len(ids)
            valid_start = max_len - valid_len
            # Keep valid_start (the <BOS> token) and the final token (<image_start>) intact
            # Replace only the prompt region between them with padding
            if valid_len > 2:
                tokens[i*2 + 1, valid_start + 1 : -1] = processor.pad_id
    else:
        tokens = input_ids_tensor
        batch_mask = attention_mask

    # =====================================================================
    # [Core fix 3] Compute exact position IDs to remove padding-induced offsets
    # =====================================================================
    position_ids = batch_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(batch_mask == 0, 1)

    # Prepare autoregressive generation
    past_key_values = DynamicCache()
    generated_tokens = torch.zeros((bs, image_token_num), dtype=torch.int, device=model.device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        inputs_embeds = model.language_model.get_input_embeddings()(tokens)
        
        # [Prefill] Pass the mask and position_ids explicitly
        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch_mask,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=past_key_values
        )
        
        hidden_states = outputs.last_hidden_state[:, -1, :]
        current_pos_ids = position_ids[:, -1:] + 1
        
        # [Autoregressive]
        for i in range(image_token_num):
            logits = model.gen_head(hidden_states)
            
            if use_cfg:
                logit_cond = logits[::2, :]
                logit_uncond = logits[1::2, :]
                logits = logit_uncond + cfg_scale * (logit_cond - logit_uncond)
            
            probs = torch.softmax(logits / temperature, dim=-1)
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                probs[indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            generated_tokens[:, i] = next_token
            
            img_embeds = model.prepare_gen_img_embeds(next_token)
            next_inputs = img_embeds.unsqueeze(1).repeat_interleave(2, dim=0) if use_cfg else img_embeds.unsqueeze(1)
            
            # Extend the attention mask dynamically
            batch_mask = torch.cat([batch_mask, torch.ones((bs_expanded, 1), dtype=torch.long, device=model.device)], dim=1)

            outputs = model.language_model.model(
                inputs_embeds=next_inputs,
                attention_mask=batch_mask,
                position_ids=current_pos_ids,
                use_cache=True,
                past_key_values=past_key_values
            )
            hidden_states = outputs.last_hidden_state[:, -1, :]
            current_pos_ids += 1

    # Decode image tokens
    dec = model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int), 
        shape=[bs, 8, img_size//patch_size, img_size//patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    
    return [Image.fromarray(img) for img in dec]


def main(args):
    rank, world_size, device = setup_ddp(args)
    
    if rank == 0:
        print(f"Sampling COCO for Janus-Pro v3 | World Size: {world_size}")

    # 1. Load Model (Stitching Mode)
    model, processor = load_stitched_janus_model(args.base_model_path, args.target_ckpt_path, device)
    
    # 2. Prepare Prompt List (Robust Reading)
    target_cols = ['caption', 'prompt', 'text', 'prompts', 'captions']
    try:
        if args.prompt_file.endswith('.tsv'):
            df = pd.read_csv(args.prompt_file, sep='\t')
        else:
            df = pd.read_csv(args.prompt_file)

        found_col = next((c for c in df.columns if c.lower() in target_cols), None)
        if found_col:
            prompt_list = df[found_col].astype(str).tolist()
        else:
            prompt_list = df.iloc[:, 0].astype(str).tolist()
    except Exception as e:
        if rank == 0:
            print(f"Standard CSV parsing failed ({str(e)}). Falling back to txt reading.")
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt_list = [line.strip() for line in f if line.strip()]

    if len(prompt_list) == 0:
        raise ValueError("Prompt list is empty!")

    # 3. Handle Sampling Quantity
    num_fid_samples = args.num_fid_samples if args.num_fid_samples > 0 else len(prompt_list)
    if len(prompt_list) < num_fid_samples:
        prompt_list = (prompt_list * (num_fid_samples // len(prompt_list) + 1))[:num_fid_samples]
    else:
        prompt_list = prompt_list[:num_fid_samples]

    # [Critical fix] Extract and write captions.txt
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True) 
        captions_path = os.path.join(args.output_dir, "captions.txt")
        print(f"Saving {len(prompt_list)} captions to {captions_path} ...")
        with open(captions_path, "w", encoding="utf-8") as f:
            for p in prompt_list:
                f.write(p.strip() + "\n")

    # 4. Distribute Work 
    indices = list(range(rank, num_fid_samples, world_size))
    local_prompts = [prompt_list[i] for i in indices]
    
    if rank == 0:
        print(f"Total Prompts: {num_fid_samples}, Prompts per GPU: {len(local_prompts)}")

    # 5. Output Directory Setup 
    images_dir = os.path.join(args.output_dir, "images")
    if rank == 0:
        os.makedirs(images_dir, exist_ok=True)
        
    if dist.is_initialized(): 
        dist.barrier()

    # 6. Generation Loop
    batch_size = args.batch_size
    iterator = range(0, len(local_prompts), batch_size)
    if rank == 0:
        iterator = tqdm(iterator, total=math.ceil(len(local_prompts)/batch_size))

    for i in iterator:
        batch_prompts = local_prompts[i : i + batch_size]
        batch_indices = indices[i : i + batch_size] 
        
        images = janus_generate(
            model, processor, batch_prompts,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            image_token_num=args.image_token_size
        )
        
        for img, global_idx in zip(images, batch_indices):
            save_path = os.path.join(images_dir, f"{global_idx:06d}.png")
            img.save(save_path)

    if dist.is_initialized(): 
        dist.barrier()

    if rank == 0:
        print(f"Generation finished. Validating output format in {images_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to original Janus Base")
    parser.add_argument("--target_ckpt_path", type=str, required=True, help="Path to SFT Checkpoint")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    parser.add_argument("--num_fid_samples", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--image_token_size", type=int, default=576)
    
    args = parser.parse_args()
    main(args)