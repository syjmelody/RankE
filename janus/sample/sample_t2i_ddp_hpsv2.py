import os
import sys
import json
import argparse
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from tqdm import tqdm
import math

from transformers import DynamicCache

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
    prompts: list,
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

    max_len = max(len(ids) for ids in input_ids_list)
    input_ids_tensor = torch.full((bs, max_len), processor.pad_id, dtype=torch.long, device=model.device)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long, device=model.device)

    for i, ids in enumerate(input_ids_list):
        input_ids_tensor[i, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=model.device)
        attention_mask[i, -len(ids):] = 1

    use_cfg = cfg_scale > 1.0
    bs_expanded = bs * 2 if use_cfg else bs

    if use_cfg:
        tokens = torch.zeros((bs_expanded, max_len), dtype=torch.long, device=model.device)
        batch_mask = torch.zeros((bs_expanded, max_len), dtype=torch.long, device=model.device)

        tokens[::2] = input_ids_tensor
        batch_mask[::2] = attention_mask

        tokens[1::2] = input_ids_tensor.clone()
        batch_mask[1::2] = attention_mask.clone()

        for i, ids in enumerate(input_ids_list):
            valid_len = len(ids)
            valid_start = max_len - valid_len
            if valid_len > 2:
                tokens[i*2 + 1, valid_start + 1 : -1] = processor.pad_id
    else:
        tokens = input_ids_tensor
        batch_mask = attention_mask

    position_ids = batch_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(batch_mask == 0, 1)

    past_key_values = DynamicCache()
    generated_tokens = torch.zeros((bs, image_token_num), dtype=torch.int, device=model.device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        inputs_embeds = model.language_model.get_input_embeddings()(tokens)

        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch_mask,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=past_key_values
        )

        hidden_states = outputs.last_hidden_state[:, -1, :]
        current_pos_ids = position_ids[:, -1:] + 1

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

    dec = model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[bs, 8, img_size // patch_size, img_size // patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    return [Image.fromarray(img) for img in dec]


def load_local_hpdv2_prompts(hpdv2_path):
    styles = ['anime', 'concept-art', 'paintings', 'photo']
    all_prompts = {}

    for style in styles:
        json_path = os.path.join(hpdv2_path, "benchmark", f"benchmark_{style}.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(hpdv2_path, "benchmark", f"{style}.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Cannot find HPDv2 prompt file: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        prompts = []
        for item in data:
            if isinstance(item, dict):
                prompts.append(item.get("prompt", ""))
            else:
                prompts.append(item)
        all_prompts[style] = prompts

    return all_prompts


def main(args):
    rank, world_size, device = setup_ddp(args)

    if rank == 0:
        print(f"HPSv2 Sampling for Janus-Pro | World Size: {world_size}")

    model, processor = load_stitched_janus_model(args.base_model_path, args.target_ckpt_path, device)

    all_prompts = load_local_hpdv2_prompts(args.hpdv2_path)
    flat_prompts = []

    for style, prompts in all_prompts.items():
        if rank == 0:
            os.makedirs(os.path.join(args.save_dir, style), exist_ok=True)
        for idx, prompt in enumerate(prompts):
            flat_prompts.append((style, idx, prompt))

    if dist.is_initialized():
        dist.barrier()

    local_prompts = flat_prompts[rank::world_size]

    if rank == 0:
        print(f"Total prompts: {len(flat_prompts)}, per GPU: {len(local_prompts)}")

    batch_size = args.batch_size
    total_steps = math.ceil(len(local_prompts) / batch_size)

    for i in tqdm(range(0, len(local_prompts), batch_size), disable=(rank != 0), total=total_steps):
        batch = local_prompts[i : i + batch_size]
        b_styles = [x[0] for x in batch]
        b_idxs = [x[1] for x in batch]
        b_texts = [x[2] for x in batch]

        # Skip already generated images
        all_exist = all(
            os.path.exists(os.path.join(args.save_dir, s, f"{idx:05d}.jpg"))
            for s, idx in zip(b_styles, b_idxs)
        )
        if all_exist:
            continue

        images = janus_generate(
            model, processor, b_texts,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            image_token_num=args.image_token_size,
        )

        for j, img in enumerate(images):
            save_path = os.path.join(args.save_dir, b_styles[j], f"{b_idxs[j]:05d}.jpg")
            img.save(save_path, format="JPEG", quality=100)

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print(">>> HPSv2 Sampling Complete!")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to original Janus Base")
    parser.add_argument("--target_ckpt_path", type=str, required=True, help="Path to SFT/RL Checkpoint")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--hpdv2-path", type=str, required=True, help="Local HPDv2 dataset path")

    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--image_token_size", type=int, default=576)

    args = parser.parse_args()
    main(args)
