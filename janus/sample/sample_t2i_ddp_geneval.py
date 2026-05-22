import os
import sys
import argparse
import json
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


def main(args):
    rank, world_size, device = setup_ddp(args)

    if rank == 0:
        print(f"GenEval Sampling for Janus-Pro | World Size: {world_size}")

    # 1. Load Model
    model, processor = load_stitched_janus_model(args.base_model_path, args.target_ckpt_path, device)

    # 2. Read GenEval JSONL
    if rank == 0:
        print(f"Loading GenEval prompts from: {args.prompts}")

    with open(args.prompts, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f.readlines() if line.strip()]

    if rank == 0:
        print(f"Total prompts: {len(all_lines)}")

    # 3. DDP Distribution (per-prompt, so all repeats of one prompt stay on one GPU)
    local_indices = list(range(rank, len(all_lines), world_size))
    local_lines = [all_lines[i] for i in local_indices]

    # 4. Batching: pack multiple prompts per forward pass
    prompts_per_batch = max(1, args.batch_size // args.repeat)

    if rank == 0:
        print(f"Prompts per GPU: {len(local_lines)}")
        print(f"Prompts per Batch: {prompts_per_batch} (generating {prompts_per_batch * args.repeat} images per step)")
        print(f"Sampling Target Directory: {args.save_dir}")

    if dist.is_initialized():
        dist.barrier()

    # 5. Generation Loop
    total_steps = (len(local_lines) + prompts_per_batch - 1) // prompts_per_batch

    for i in tqdm(range(0, len(local_lines), prompts_per_batch), disable=(rank != 0), total=total_steps):
        batch_lines = local_lines[i : i + prompts_per_batch]
        batch_global_indices = local_indices[i : i + prompts_per_batch]

        flat_prompts = []
        meta_infos = []

        for line, g_idx in zip(batch_lines, batch_global_indices):
            metadata = json.loads(line)
            p_text = metadata['prompt']

            flat_prompts.extend([p_text] * args.repeat)

            for r in range(args.repeat):
                meta_infos.append((g_idx, r, line))

        images = janus_generate(
            model, processor, flat_prompts,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_k=args.top_k,
            image_token_num=args.image_token_size,
            img_size=args.image_size,
            patch_size=args.patch_size,
        )

        for img_idx, img in enumerate(images):
            g_idx, sub_idx, line_str = meta_infos[img_idx]

            prompt_dir = os.path.join(args.save_dir, str(g_idx).zfill(5))
            sample_dir = os.path.join(prompt_dir, "samples")

            if sub_idx == 0:
                os.makedirs(sample_dir, exist_ok=True)
                meta_path = os.path.join(prompt_dir, "metadata.jsonl")
                if not os.path.exists(meta_path):
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        f.write(line_str + "\n")

            save_path = os.path.join(sample_dir, f"{str(sub_idx).zfill(5)}.png")
            img.save(save_path)

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print("GenEval Sampling Finished.")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--prompts", type=str, required=True, help="GenEval evaluation_metadata.jsonl")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to original Janus Base")
    parser.add_argument("--target_ckpt_path", type=str, required=True, help="Path to SFT/RL Checkpoint")

    # Inference
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=4, help="Samples per prompt for GenEval")
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16, help="Total images per GPU per step")
    parser.add_argument("--image_token_size", type=int, default=576)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=16)

    args = parser.parse_args()
    main(args)
