import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import copy
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from accelerate import Accelerator
from PIL import Image
from collections import defaultdict, deque

from transformers import DynamicCache

# --- LlamaGen-specific dependencies (safe as long as Janus mode does not call them) ---
try:
    from autoregressive.models.generate import generate as llamagen_generate
except ImportError:
    pass
from autoregressive.models.vqrl_models import HybridVQLoss, MultiRewardManager

@dataclass
class HybridTrainConfig:
    """
    Unified Hybrid RL Framework Configuration
    Compatible with both LlamaGen and Janus-Pro, using a LlamaGen-aligned config surface.
    """
    model_type: str = "llamagen"
    train_mode: str = "both"
    
    # --- Optimization ---
    learning_rate_gpt: float = 1e-6       
    learning_rate_decoder: float = 1e-5
    learning_rate_disc: float = 2e-5
    learning_rate_codebook: float = 5e-7
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    
    # --- GRPO (Stream 1) ---
    group_size: int = 4
    grpo_epochs: int = 1
    rollout_length: int = 256
    clip_eps: float = 0.2
    kl_coef: float = 0.05
    grpo_adv_coef: float = 5.0
    
    # --- RankE / Decoder (Stream 2) ---
    lambda_decoder_reward: float = 0.05
    lambda_decoder_gan: float = 0.1
    lambda_lasc_consistency: float = 1.0
    lambda_codebook_anchor: float = 10.0
    lambda_reconstruction: float = 1.0
    lambda_decoder_consistency: float = 5.0
    
    # Decoder Consistency Scheduling
    consistency_schedule_type: str = "none"
    consistency_start_step: int = 0
    consistency_end_step: int = 1000
    consistency_start_value: float = 0.0
    consistency_end_value: float = 5.0
    
    rejection_sample_k: int = 2
    lasc_sample_k: int = 20
    decoder_topk_rewards: int = 1
    
    # Annealing parameters (dynamic temperature)
    decoder_resample_temp: float = 1.0      
    decoder_resample_temp_start: float = 4.0
    decoder_resample_temp_end: float = 1.0
    anneal_ratio: float = 0.1
    
    # --- Discriminator ---
    disc_start: int = 0
    disc_weight: float = 0.5
    disc_type: str = "projected"
    dino_path: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    gen_adv_loss: str = 'hinge'
    
    # --- Model Specs ---
    codebook_embed_dim: int = 8
    image_size: int = 512
    batch_size: int = 1
    
    # --- Generation ---
    gen_cfg_scale: float = 7.5
    gen_top_k: int = 1000
    gen_temperature: float = 1.0
    
    # --- EMA ---
    ema_decay_vq: float = 0.995
    ema_decay_gpt: float = 0.99
    frqs_ema_gpt_update: int = 10
    frqs_ema_vq_update: int = 2

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            if k == 'lr_gpt': self.learning_rate_gpt = v
            if k == 'lr_decoder': self.learning_rate_decoder = v
            if k == 'lr_disc': self.learning_rate_disc = v
            if k == 'temp_start': self.decoder_resample_temp_start = v
            if k == 'temp_end': self.decoder_resample_temp_end = v


class UnifiedHybridRLTrainer:
    """
    Unified hybrid trainer that smooths over architecture differences via an adapter-style design
    Closely follows the LlamaGen HybridRLTrainer logic while swapping the model interface layer
    """
    def __init__(
        self,
        gpt_model,
        vq_model,
        reward_model: MultiRewardManager,
        config: HybridTrainConfig,
        accelerator: Accelerator,
        t5_model = None,
        processor = None,
        ema_gpt_model = None,
        ema_vq_model = None,
        ref_gpt_model = None,
    ):
        self.gpt_model = gpt_model
        self.vq_model = vq_model
        self.reward_model = reward_model
        self.t5_model = t5_model
        self.processor = processor
        self.config = config
        self.accelerator = accelerator
        
        # ==================== 1. Model state setup ====================
        self.vq_model.requires_grad_(False)
        if self.config.train_mode in ['decoder', 'both']:
            if hasattr(self.vq_model, 'decoder'):
                self.vq_model.decoder.requires_grad_(True)
                self.vq_model.decoder.train()
            if hasattr(self.vq_model, 'post_quant_conv'):
                self.vq_model.post_quant_conv.requires_grad_(True)
        
        if self.config.train_mode == 'decoder':
            self.gpt_model.requires_grad_(False).eval()
        else:
            self.gpt_model.train()

        # ==================== 2. Initialize hybrid VQ loss ====================
        self.vq_loss = HybridVQLoss(
            disc_start=config.disc_start,
            disc_weight=config.disc_weight,
            disc_type=config.disc_type,
            dino_path=config.dino_path,
            perceptual_weight=1.0,
            reconstruction_weight=config.lambda_reconstruction
        ).to(self.accelerator.device)
        
        # ==================== 3. Freeze static models ====================
        if self.t5_model is not None:
            self.t5_model.model.requires_grad_(False)
            
        # ==================== 4. EMA models ====================
        self.ema_gpt_model = ema_gpt_model
        self.ema_vq_model = ema_vq_model
        self.ref_gpt_model = ref_gpt_model

        # ==================== 5. Optimizer setup ====================
        self.gpt_optimizer = None
        self.decoder_optimizer = None
        self.disc_optimizer = None

        if self.config.train_mode in ['gpt', 'both']:
            gpt_params = []
            if self.config.model_type == 'janus-pro' and self.config.train_mode == 'both':
                vq_param_ids = {id(p) for p in self.vq_model.parameters()}
                gpt_params = [p for p in self.gpt_model.parameters() if p.requires_grad and id(p) not in vq_param_ids]
            else:
                gpt_params = list(filter(lambda p: p.requires_grad, self.gpt_model.parameters()))
                
            self.gpt_optimizer = optim.AdamW(
                gpt_params,
                lr=config.learning_rate_gpt, weight_decay=config.weight_decay
            )
        
        if self.config.train_mode in ['decoder', 'both']:
            dec_params = list(filter(lambda p: p.requires_grad, self.vq_model.parameters()))
            self.decoder_optimizer = optim.AdamW(
                dec_params, lr=config.learning_rate_decoder, weight_decay=config.weight_decay
            )
            self.disc_optimizer = optim.AdamW(
                self.vq_loss.discriminator.parameters(),
                lr=config.learning_rate_disc, weight_decay=config.weight_decay
            )

        # ==================== 6. Accelerator Prepare ====================
        if self.gpt_optimizer:
            if self.config.model_type == 'llamagen':
                self.gpt_model, self.gpt_optimizer = accelerator.prepare(self.gpt_model, self.gpt_optimizer)
            else:
                # Janus: forward passes bypass the DDP wrapper, so only the optimizer is prepared
                self.gpt_optimizer = accelerator.prepare(self.gpt_optimizer)
        else:
            self.gpt_model.to(accelerator.device)
            
        if self.decoder_optimizer:
            if self.config.model_type == 'llamagen':
                self.vq_model, self.decoder_optimizer, self.disc_optimizer, self.vq_loss = accelerator.prepare(
                    self.vq_model, self.decoder_optimizer, self.disc_optimizer, self.vq_loss
                )
            else:
                # Janus: VQ lives inside GPT, so it is not wrapped independently; vq_loss also stays outside DDP to avoid "ready twice" errors
                self.decoder_optimizer, self.disc_optimizer = accelerator.prepare(
                    self.decoder_optimizer, self.disc_optimizer
                )
                self.vq_loss.to(accelerator.device)
        else:
            self.vq_model.to(accelerator.device)
            self.vq_loss.to(accelerator.device)

        # ==================== 7. Counter initialization (aligned with LlamaGen) ====================
        self.counter_gpt_update = 0
        self.counter_vq_update = 0
        self.counter_disc_update = 0
        self.counter_ema_gpt_update = 0
        self.counter_ema_vq_update = 0
        
        self.freqs_ema_vq_update: int = config.frqs_ema_vq_update
        self.freqs_ema_gpt_update: int = config.frqs_ema_gpt_update
            
        self.register_buffer_stats()

    def register_buffer_stats(self):
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.accelerator.device)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.accelerator.device)

    def toggle_requires_grad(self, model, requires_grad: bool):
        for param in model.parameters():
            param.requires_grad = requires_grad

    # ==================== Helper Functions ====================
    def clear_kv_cache(self, model):
        unwrapped = self.accelerator.unwrap_model(model)
        if hasattr(unwrapped, 'layers'):
            for layer in unwrapped.layers:
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'kv_cache'):
                    layer.attention.kv_cache = None
        torch.cuda.empty_cache()

    def _get_current_lr(self, optimizer):
        if optimizer is not None and len(optimizer.param_groups) > 0:
            return optimizer.param_groups[0]['lr']
        return 0.0

    def get_scheduled_consistency_lambda(self, global_step: int) -> float:
        cfg = self.config
        if cfg.consistency_schedule_type == "none":
            return cfg.lambda_decoder_consistency

        start_step = cfg.consistency_start_step
        end_step = cfg.consistency_end_step
        start_val = cfg.consistency_start_value
        end_val = cfg.consistency_end_value

        if global_step <= start_step:
            return start_val
        if global_step >= end_step:
            return end_val

        progress = (global_step - start_step) / max(end_step - start_step, 1)

        if cfg.consistency_schedule_type == "linear":
            return start_val + (end_val - start_val) * progress
        elif cfg.consistency_schedule_type == "sin":
            return end_val + (start_val - end_val) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            return cfg.lambda_decoder_consistency

    def update_ema_model(self, types: List[str]):
        with torch.no_grad():
            if 'gpt' in types and self.ema_gpt_model is not None:
                current_gpt = self.accelerator.unwrap_model(self.gpt_model)
                decay = self.config.ema_decay_gpt
                for param_q, param_k in zip(current_gpt.parameters(), self.ema_gpt_model.parameters()):
                    param_k.data.mul_(decay).add_(param_q.data, alpha=1 - decay)
            if 'vq' in types and self.ema_vq_model is not None:
                current_vq = self.accelerator.unwrap_model(self.vq_model)
                decay = self.config.ema_decay_vq
                for param_q, param_k in zip(current_vq.parameters(), self.ema_vq_model.parameters()):
                    param_k.data.mul_(decay).add_(param_q.data, alpha=1 - decay)

    def compute_differentiable_clip_score(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        norm_images = (images - self.clip_mean) / self.clip_std
        if norm_images.shape[-1] != 224:
            norm_images = F.interpolate(norm_images, size=(224, 224), mode='bilinear', align_corners=False)
        
        image_features = self.reward_model.clip_model.get_image_features(pixel_values=norm_images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        with torch.no_grad():
            text_inputs = self.reward_model.clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.accelerator.device)
            text_features = self.reward_model.clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        if len(texts) != images.shape[0]:
            repeat_factor = images.shape[0] // len(texts)
            text_features = text_features.repeat_interleave(repeat_factor, dim=0)
            
        scores = (image_features * text_features).sum(dim=-1)
        return scores * 100.0

    def compute_grpo_advantages(self, rewards: torch.Tensor, group_ids: torch.Tensor):
        adv = torch.zeros_like(rewards)
        unique_groups = group_ids.unique()
        for g in unique_groups:
            mask = (group_ids == g)
            grp_r = rewards[mask]
            if grp_r.numel() > 1:
                mean_r = grp_r.mean()
                std_r = grp_r.std()
                if std_r > 1e-4:
                    adv[mask] = (grp_r - mean_r) / std_r
                else:
                    adv[mask] = 0.0
            else:
                adv[mask] = grp_r 
        return torch.clamp(adv, -5.0, 5.0)

    # =========================================================================
    # [Model adapter 1] Build text conditions (embeddings / token IDs)
    # =========================================================================
    def prepare_conditions(self, prompts: List[str], device):
        if self.config.model_type == "llamagen":
            embs, masks = self.t5_model.get_text_embeddings(prompts)
            return {"cond_idx": embs, "mask": masks}
            
        elif self.config.model_type == "janus-pro":
            bs = len(prompts)
            input_ids_list = []
            for prompt in prompts:
                conv = [{"role": "<|User|>", "content": prompt}, {"role": "<|Assistant|>", "content": ""}]
                sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
                    conversations=conv, sft_format=self.processor.sft_format, system_prompt=""
                )
                full_prompt = sft_format + self.processor.image_start_tag
                input_ids_list.append(self.processor.tokenizer.encode(full_prompt))

            max_len = max(len(ids) for ids in input_ids_list)
            input_ids_tensor = torch.full((bs, max_len), self.processor.pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros((bs, max_len), dtype=torch.long, device=device)

            for i, ids in enumerate(input_ids_list):
                input_ids_tensor[i, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=device)
                attention_mask[i, -len(ids):] = 1

            return {"input_ids": input_ids_tensor, "attention_mask": attention_mask}

    # =========================================================================
    # [Model adapter 2] Get token log-probabilities (forward pass)
    # =========================================================================
    def get_per_token_logps(self, model, conds, sequences):
        was_training = model.training
        model.eval()
        unwrapped_model = self.accelerator.unwrap_model(model)
        self.clear_kv_cache(model)

        if self.config.gradient_checkpointing:
            if hasattr(unwrapped_model, "language_model") and hasattr(unwrapped_model.language_model, "gradient_checkpointing_enable"):
                unwrapped_model.language_model.gradient_checkpointing_enable()
            elif hasattr(unwrapped_model, "gradient_checkpointing_enable"):
                unwrapped_model.gradient_checkpointing_enable()

        B, L_img = sequences.shape
        device = sequences.device
        
        # ----------------- [LlamaGen Route] -----------------
        if self.config.model_type == "llamagen":
            text_embs, emb_masks = conds["cond_idx"], conds["mask"]
            T_text = text_embs.shape[1]
            input_ids = sequences[:, :-1]
            targets = sequences
            
            total_len = T_text + input_ids.shape[1]
            input_pos = torch.arange(0, total_len, device=device, dtype=torch.long)
            full_mask = torch.tril(torch.ones((1, 1, total_len, total_len), device=device, dtype=torch.bool))
            full_mask[:, :, :T_text, :T_text] = True 
            full_mask = full_mask.expand(B, -1, -1, -1).clone()

            if emb_masks is not None:
                 valid_text = emb_masks.view(B, 1, 1, T_text).bool()
                 valid_image = torch.ones((B, 1, 1, input_ids.shape[1]), device=device, dtype=torch.bool)
                 key_mask = torch.cat([valid_text, valid_image], dim=-1)
                 full_mask = full_mask & key_mask

            logits, _ = model(idx=input_ids.long(), cond_idx=text_embs, input_pos=input_pos, mask=full_mask)
            image_logits = logits[:, T_text - 1 :, :]
            
        # ----------------- [Janus-Pro Route] -----------------
        elif self.config.model_type == "janus-pro":
            input_ids, attention_mask = conds["input_ids"], conds["attention_mask"]
            T_text = input_ids.shape[1]
            
            img_embeds = unwrapped_model.prepare_gen_img_embeds(sequences[:, :-1])
            prompt_embeds = unwrapped_model.language_model.get_input_embeddings()(input_ids)
            
            inputs_embeds = torch.cat([prompt_embeds, img_embeds], dim=1)
            img_mask = torch.ones((B, img_embeds.shape[1]), dtype=torch.long, device=device)
            full_mask = torch.cat([attention_mask, img_mask], dim=1)
            
            pos_ids_prompt = attention_mask.long().cumsum(-1) - 1
            pos_ids_prompt.masked_fill_(attention_mask == 0, 1)
            pos_ids_img = pos_ids_prompt[:, -1:] + torch.arange(1, img_embeds.shape[1] + 1, device=device).unsqueeze(0)
            full_pos_ids = torch.cat([pos_ids_prompt, pos_ids_img], dim=1)

            outputs = unwrapped_model.language_model.model(
                inputs_embeds=inputs_embeds, 
                attention_mask=full_mask, 
                position_ids=full_pos_ids,
                use_cache=False
            )
            logits = unwrapped_model.gen_head(outputs.last_hidden_state)
            image_logits = logits[:, T_text - 1 :, :]

        # --- Unified NLL computation ---
        logits_flat = image_logits.reshape(-1, image_logits.size(-1))
        targets_flat = sequences.reshape(-1).long()
        nll_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        token_lp = -nll_loss.view(B, sequences.shape[1])

        self.clear_kv_cache(model)
        if was_training: model.train()
        
        return token_lp, image_logits

    # =========================================================================
    # [Model adapter 3] Autoregressive generation
    # =========================================================================
    @torch.no_grad()
    def core_generate(self, model, conds, batch_prompts):
        unwrapped_model = self.accelerator.unwrap_model(model)
        
        if self.config.model_type == "llamagen":
            return llamagen_generate(
                unwrapped_model, conds["cond_idx"], self.config.rollout_length, 
                emb_masks=conds["mask"], cfg_scale=self.config.gen_cfg_scale, 
                temperature=self.config.gen_temperature, top_k=self.config.gen_top_k, sample_logits=True
            )
            
        elif self.config.model_type == "janus-pro":
            input_ids_raw = conds["input_ids"]
            attention_mask_raw = conds["attention_mask"] 
            B = input_ids_raw.shape[0]
            
            start_token_mask = (input_ids_raw == self.processor.image_start_id)
            max_prompt_len = start_token_mask.int().argmax(dim=1).max().item() + 1 if start_token_mask.any() else input_ids_raw.shape[1]
            
            cond_ids = input_ids_raw[:, :max_prompt_len]
            cond_mask = attention_mask_raw[:, :max_prompt_len]
            position_ids = cond_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(cond_mask == 0, 1)
            
            use_cfg = self.config.gen_cfg_scale > 1.0
            bs_expanded = B * 2 if use_cfg else B
            
            if use_cfg:
                tokens = torch.zeros((bs_expanded, max_prompt_len), dtype=torch.long, device=self.accelerator.device)
                batch_mask = torch.zeros((bs_expanded, max_prompt_len), dtype=torch.long, device=self.accelerator.device)
                batch_pos_ids = torch.zeros((bs_expanded, max_prompt_len), dtype=torch.long, device=self.accelerator.device)
                
                tokens[::2], batch_mask[::2], batch_pos_ids[::2] = cond_ids, cond_mask, position_ids
                tokens[1::2], batch_mask[1::2], batch_pos_ids[1::2] = cond_ids.clone(), cond_mask.clone(), position_ids.clone()
                for i in range(B):
                    nonzero = cond_mask[i].nonzero()
                    if len(nonzero) > 0:
                        start_idx, end_idx = nonzero[0].item(), max_prompt_len - 1
                        if end_idx > start_idx + 1:
                            tokens[i*2+1, start_idx+1 : end_idx] = self.processor.pad_id
            else:
                tokens, batch_mask, batch_pos_ids = cond_ids, cond_mask, position_ids

            past_key_values = DynamicCache()
            generated_tokens = torch.zeros((B, self.config.rollout_length), dtype=torch.int, device=self.accelerator.device)

            with self.accelerator.autocast():
                inputs_embeds = unwrapped_model.language_model.get_input_embeddings()(tokens)
                outputs = unwrapped_model.language_model.model(
                    inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values, 
                    attention_mask=batch_mask, position_ids=batch_pos_ids
                )
                
                hidden_states = outputs.last_hidden_state[:, -1, :]
                current_pos_ids = batch_pos_ids[:, -1:] + 1
                
                for i in range(self.config.rollout_length):
                    logits = unwrapped_model.gen_head(hidden_states)
                    if use_cfg:
                        logit_cond, logit_uncond = logits[::2, :], logits[1::2, :]
                        logits = logit_uncond + self.config.gen_cfg_scale * (logit_cond - logit_uncond)
                    
                    probs = torch.softmax(logits / self.config.gen_temperature, dim=-1)
                    if self.config.gen_top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, self.config.gen_top_k)[0][..., -1, None]
                        probs[indices_to_remove] = 0
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                        
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    generated_tokens[:, i] = next_token
                    
                    img_embeds = unwrapped_model.prepare_gen_img_embeds(next_token)
                    next_inputs = img_embeds.unsqueeze(1).repeat_interleave(2, dim=0) if use_cfg else img_embeds.unsqueeze(1)
                    batch_mask = torch.cat([batch_mask, torch.ones((bs_expanded, 1), dtype=torch.long, device=self.accelerator.device)], dim=1)
                    
                    outputs = unwrapped_model.language_model.model(
                        inputs_embeds=next_inputs, use_cache=True, past_key_values=past_key_values, 
                        attention_mask=batch_mask, position_ids=current_pos_ids
                    )
                    hidden_states = outputs.last_hidden_state[:, -1, :]
                    current_pos_ids += 1
            return generated_tokens

    # =========================================================================
    # [Core flow] Step 1: rollout and rewards (aligned with LlamaGen)
    # =========================================================================
    @torch.no_grad()
    def generate_rollouts(self, batch_prompts: List[str]):
        self.gpt_model.eval()
        B, G = len(batch_prompts), self.config.group_size
        
        # 1. Prepare and expand conditions
        base_conds = self.prepare_conditions(batch_prompts, self.accelerator.device)
        exp_conds = {k: v.repeat_interleave(G, dim=0) for k, v in base_conds.items()}
        expanded_prompts = [p for p in batch_prompts for _ in range(G)]
        
        # 2. Generate samples
        with self.accelerator.autocast():
            sequences = self.core_generate(self.gpt_model, exp_conds, expanded_prompts)
        if not sequences.is_contiguous(): sequences = sequences.contiguous()
        
        self.clear_kv_cache(self.gpt_model)
        if self.ema_gpt_model: self.clear_kv_cache(self.ema_gpt_model)
        
        # 3. Compute Log Probs
        with self.accelerator.autocast():
            old_log_probs, full_logits = self.get_per_token_logps(self.gpt_model, exp_conds, sequences)
            
            ref_model_to_use = self.ref_gpt_model if self.ref_gpt_model is not None else self.ema_gpt_model
            if ref_model_to_use is not None:
                ref_log_probs, _ = self.get_per_token_logps(ref_model_to_use, exp_conds, sequences)
            else:
                ref_log_probs = old_log_probs.clone()
                
        # 4. Compute rewards using EMA VQ decoding for stable reward signals
        B_total, L_seq = sequences.shape
        h = w = int(math.sqrt(L_seq))
        vq_for_reward = self.ema_vq_model if self.ema_vq_model is not None else self.vq_model
        unwrapped_vq = self.accelerator.unwrap_model(vq_for_reward)
        unwrapped_vq.eval()
        
        z_q = unwrapped_vq.quantize.get_codebook_entry(
            sequences.view(-1), shape=(B_total, h, w, self.config.codebook_embed_dim), channel_first=False
        ).permute(0, 3, 1, 2)
        
        images = unwrapped_vq.decode(z_q)
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)
            
        final_rewards, reward_breakdown = self.reward_model(images, expanded_prompts)
        
        # 5. Store Top-K Logits for LASC (CPU Offload to save VRAM)
        K_lasc = self.config.lasc_sample_k
        top_k_vals, top_k_inds = torch.topk(full_logits, K_lasc, dim=-1)
        packed_logits = {
            "values": top_k_vals.detach().cpu(),
            "indices": top_k_inds.detach().cpu()
        }
        
        if self.config.train_mode in ['gpt', 'both']:
            self.gpt_model.train()
            
        return {
            "prompts": batch_prompts, "conds": exp_conds,
            "sequences": sequences.detach(),
            "old_log_probs": old_log_probs.detach(),
            "ref_log_probs": ref_log_probs.detach(),
            "rewards": final_rewards.to(self.accelerator.device).detach(),
            "reward_breakdown": {k: v.detach() if torch.is_tensor(v) else v for k, v in reward_breakdown.items()},
            "packed_logits": packed_logits,
            "group_ids": torch.arange(B, device=self.accelerator.device).repeat_interleave(G)
        }

    # =========================================================================
    # [Core flow] Step 2: GPT policy optimization (aligned with LlamaGen)
    # =========================================================================
    def update_gpt_policy(self, data: Dict, glob_step: int):
        conds = data["conds"]
        sequences = data["sequences"]
        old_log_probs = data["old_log_probs"].detach()
        ref_log_probs = data["ref_log_probs"].to(self.accelerator.device).detach()
        rewards = data["rewards"]
        reward_breakdown = data.get("reward_breakdown", {})
        group_ids = data["group_ids"]

        advantages = self.compute_grpo_advantages(rewards, group_ids).detach()

        self.gpt_model.train()
        unwrapped_gpt = self.accelerator.unwrap_model(self.gpt_model)
        if self.config.gradient_checkpointing:
            if hasattr(unwrapped_gpt, "language_model") and hasattr(unwrapped_gpt.language_model, "gradient_checkpointing_enable"):
                unwrapped_gpt.language_model.gradient_checkpointing_enable()
            elif hasattr(unwrapped_gpt, "gradient_checkpointing_enable"):
                unwrapped_gpt.gradient_checkpointing_enable()

        metrics_accum = defaultdict(list)
        valid_mask = torch.ones_like(sequences, dtype=torch.float32, device=self.accelerator.device)

        for i_epoch in range(self.config.grpo_epochs):
            self.gpt_optimizer.zero_grad() 
            new_log_probs, _ = self.get_per_token_logps(self.gpt_model, conds, sequences)

            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            
            adv_expanded = advantages.unsqueeze(1).expand_as(ratio)
            surr1 = ratio * adv_expanded
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * adv_expanded
            policy_loss = -torch.min(surr1, surr2).mean()

            kl_div = 0.5 * (new_log_probs - ref_log_probs)**2
            kl_loss = (kl_div * valid_mask).sum() / (valid_mask.sum() + 1e-8)

            total_loss = policy_loss + self.config.kl_coef * kl_loss
            
            self.accelerator.backward(total_loss)
            if self.accelerator.sync_gradients:
                # Janus: forward bypasses DDP, so gradients must be all-reduced manually
                if self.config.model_type == 'janus-pro' and self.accelerator.num_processes > 1:
                    for param_group in self.gpt_optimizer.param_groups:
                        for param in param_group['params']:
                            if param.grad is not None:
                                torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
                                param.grad.data.div_(self.accelerator.num_processes)
                if self.config.max_grad_norm > 0:
                    if self.config.model_type == 'llamagen':
                        self.accelerator.clip_grad_norm_(self.gpt_model.parameters(), self.config.max_grad_norm)
                    else:
                        params_to_clip = [p for group in self.gpt_optimizer.param_groups for p in group['params']]
                        self.accelerator.clip_grad_norm_(params_to_clip, self.config.max_grad_norm)
                self.gpt_optimizer.step()
                self.counter_gpt_update += 1
                
                if self.counter_gpt_update % self.freqs_ema_gpt_update == 0:
                    self.update_ema_model(['gpt'])
                    self.counter_ema_gpt_update += 1
            
            metrics_accum["loss_gpt/surr1_ratio"].append(ratio.mean().item())
            metrics_accum["loss_gpt/surr1_adv"].append(adv_expanded.mean().item())
            metrics_accum["loss_gpt/policy_surr1"].append(surr1.mean().item())
            metrics_accum["loss_gpt/policy_surr2"].append(surr2.mean().item())
            metrics_accum["loss_gpt/policy_loss"].append(policy_loss.item())
            metrics_accum["loss_gpt/policy_kl"].append(kl_loss.item())

        avg_logs = {k: sum(v)/len(v) if v else 0.0 for k, v in metrics_accum.items()}
        
        avg_logs["reward/total_mean"] = rewards.mean().item()
        rewards_reshaped = rewards.view(-1, self.config.group_size)
        avg_logs["reward/total_std"] = rewards_reshaped.std(dim=1).mean().item()
        
        for k, v in reward_breakdown.items():
            if isinstance(v, torch.Tensor):
                avg_logs[f"reward/raw_{k}"] = v.float().mean().item()
            else:
                avg_logs[f"reward/raw_{k}"] = v

        return avg_logs, rewards

    # =========================================================================
    # [Core flow] Step 3: decoder update (LlamaGen-aligned RankE + LASC + FP32 anchor)
    # =========================================================================
    def update_decoder(self, rollout_data: Dict, gt_images: torch.Tensor, global_step: int):
        self.vq_model.train()
        
        unwrapped_vq = self.accelerator.unwrap_model(self.vq_model)
        unwrapped_vq_loss = self.accelerator.unwrap_model(self.vq_loss)
        
        metrics = {}
        device = self.accelerator.device
        
        if self.config.gradient_checkpointing:
            if hasattr(unwrapped_vq.decoder, "gradient_checkpointing_enable"):
                unwrapped_vq.decoder.gradient_checkpointing_enable()
            elif hasattr(unwrapped_vq.decoder, "gradient_checkpointing"): 
                unwrapped_vq.decoder.gradient_checkpointing = True

        # --- RankE Weights Calculation ---
        rewards = rollout_data["rewards"]
        B, G = len(rollout_data["prompts"]), self.config.group_size
        K = min(self.config.rejection_sample_k, G)
        
        rewards_reshaped = rewards.view(B, G)
        sorted_vals, sorted_indices = torch.sort(rewards_reshaped, dim=1, descending=True)
        top_k_indices = sorted_indices[:, :K]
        top_k_rewards = sorted_vals[:, :K]
        
        weights = F.softmax(top_k_rewards / self.config.decoder_resample_temp, dim=1).detach()
        weights_flat = weights.view(-1)
        weights_flat = weights_flat / (weights_flat.mean() + 1e-8)
        
        # ---------------------------------------------------------
        # 1. Data preparation (no grad, FP32 anchor)
        # ---------------------------------------------------------
        with torch.no_grad():
            # A. GT Encoding
            if self.config.model_type == 'llamagen':
                # LlamaGen VQ uses FP32 weights, so autocast is disabled for the FP32 anchor path
                with torch.cuda.amp.autocast(enabled=False):
                    z_q_gt, _, _ = unwrapped_vq.encode(gt_images.float())
            else:
                # Janus VQ uses BF16 weights, so autocast keeps dtypes consistent
                with self.accelerator.autocast():
                    z_q_gt, _, _ = unwrapped_vq.encode(gt_images)
            
            # B. Gather Top-K Sequences
            sequences = rollout_data["sequences"].detach().reshape(B, G, -1)
            L_seq = sequences.shape[-1]
            expanded_indices = top_k_indices.unsqueeze(-1).expand(B, K, L_seq)
            selected_sequences = torch.gather(sequences, 1, expanded_indices).reshape(-1, L_seq)
            
            B_sel = selected_sequences.shape[0]
            h = w = int(math.sqrt(L_seq))
            if self.config.model_type == 'llamagen':
                with torch.cuda.amp.autocast(enabled=False):
                    z_q_policy = unwrapped_vq.quantize.get_codebook_entry(
                        selected_sequences.view(-1), shape=(B_sel, h, w, self.config.codebook_embed_dim), channel_first=False
                    ).permute(0, 3, 1, 2).float()
            else:
                with self.accelerator.autocast():
                    z_q_policy = unwrapped_vq.quantize.get_codebook_entry(
                        selected_sequences.view(-1), shape=(B_sel, h, w, self.config.codebook_embed_dim), channel_first=False
                    ).permute(0, 3, 1, 2)
                z_q_policy = z_q_policy.float()

            batch_prompts = rollout_data["prompts"]
            target_prompts = [p for p in batch_prompts for _ in range(K)]
            
            del sequences, expanded_indices, selected_sequences

        # =====================================================================
        # Stage 1: update the discriminator
        # =====================================================================
        self.toggle_requires_grad(unwrapped_vq_loss.discriminator, True)
        self.disc_optimizer.zero_grad()

        with torch.no_grad():
            with self.accelerator.autocast():
                rec_gt_detached = unwrapped_vq.decode(z_q_gt).detach()

        loss_disc, _ = self.vq_loss(
            inputs=gt_images.float(), 
            reconstructions=rec_gt_detached.float(), 
            optimizer_idx=1, global_step=global_step, last_layer=None
        )
        self.accelerator.backward(loss_disc)
        
        # In Janus mode, all-reduce discriminator gradients manually
        if self.config.model_type == 'janus-pro' and self.accelerator.num_processes > 1:
            for param_group in self.disc_optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
                        param.grad.data.div_(self.accelerator.num_processes)
        
        self.disc_optimizer.step()
        self.counter_disc_update += 1
        metrics["loss_dec/disc_loss"] = loss_disc.item()

        # =====================================================================
        # Stage 2: update the generator/decoder
        # =====================================================================
        self.toggle_requires_grad(unwrapped_vq_loss.discriminator, False)
        
        self.decoder_optimizer.zero_grad()
        last_layer = unwrapped_vq.decoder.last_layer if hasattr(unwrapped_vq.decoder, "last_layer") else None

        # --- Stream A: GT Recon ---
        gt_loss_item = 0.0
        if self.config.lambda_reconstruction > 0:
            with self.accelerator.autocast():
                rec_gt = unwrapped_vq.decode(z_q_gt)
            
            loss_gen_gt, _ = self.vq_loss(
                inputs=gt_images.float(), reconstructions=rec_gt.float(),
                optimizer_idx=0, global_step=global_step, last_layer=last_layer
            )
            loss_gen_gt = loss_gen_gt * self.config.lambda_reconstruction
            self.accelerator.backward(loss_gen_gt)
            gt_loss_item = loss_gen_gt.item()

        # --- Stream B: Policy Optimization (RankE Weighted GAN & Rewards) ---
        with self.accelerator.autocast():
            rec_policy = unwrapped_vq.decode(z_q_policy.requires_grad_(True))
        rec_policy = torch.clamp(rec_policy, -1.0, 1.0).float()

        loss_gan_fake = torch.tensor(0., device=device)
        if global_step >= self.config.disc_start:
            with self.accelerator.autocast():
                logits_fake = unwrapped_vq_loss.discriminator(rec_policy)
                
            if isinstance(logits_fake, list):
                 l_acc = 0
                 for l in logits_fake:
                     w_broad = weights_flat.view(-1, 1, 1, 1) if l.dim() == 4 else weights_flat.view(-1, 1)
                     l_acc += -torch.mean(l.float() * w_broad)
                 loss_gan_fake = l_acc / len(logits_fake)
            else:
                 w_broad = weights_flat.view(-1, 1, 1, 1) if logits_fake.dim() == 4 else weights_flat.view(-1, 1)
                 loss_gan_fake = -torch.mean(logits_fake.float() * w_broad)
            loss_gan_fake = loss_gan_fake * self.config.lambda_decoder_gan * self.config.disc_weight

        # 3. Perception Consistency (EMA)
        loss_consist = torch.tensor(0., device=device)
        if self.config.lambda_decoder_consistency > 0:
            with torch.no_grad():
                self.ema_vq_model.eval()
                with self.accelerator.autocast():
                    ema_targets = self.ema_vq_model.decode(z_q_policy)
                    ema_targets = torch.clamp(ema_targets, -1.0, 1.0)
            
            p_loss = unwrapped_vq_loss.perceptual_loss(rec_policy, ema_targets.float()).mean()
            loss_consist = p_loss * self.config.lambda_decoder_consistency
            del ema_targets

        # 4. Direct Reward Loss (RankE Weighted)
        loss_reward = torch.tensor(0., device=device)
        if self.config.lambda_decoder_reward > 0:
            rec_norm = torch.clamp((rec_policy + 1.0) * 0.5, 0.0, 1.0)
            rewards_diff, _ = self.reward_model(rec_norm, target_prompts)
            loss_reward = -(rewards_diff * weights_flat).mean() * self.config.lambda_decoder_reward

        # >>> Stream C: LASC <<<
        loss_lasc = torch.tensor(0.0, device=device)
        if self.config.lambda_lasc_consistency > 0:
            packed_vals = rollout_data["packed_logits"]["values"].to(device, non_blocking=True)
            packed_inds = rollout_data["packed_logits"]["indices"].to(device, non_blocking=True)
            K_sample = packed_vals.shape[-1]
            logits_val = packed_vals.view(B, G, L_seq, K_sample)
            logits_ind = packed_inds.view(B, G, L_seq, K_sample)
            
            gather_idx = top_k_indices.view(B, K, 1, 1).expand(-1, -1, L_seq, K_sample)
            sel_logits_val = torch.gather(logits_val, 1, gather_idx).reshape(-1, L_seq, K_sample)
            sel_logits_ind = torch.gather(logits_ind, 1, gather_idx).reshape(-1, L_seq, K_sample)
            
            with torch.cuda.amp.autocast(enabled=False):
                soft_probs = F.softmax(sel_logits_val.float(), dim=-1)
                codebook = unwrapped_vq.quantize.embedding.weight.float() 
                soft_vecs = F.embedding(sel_logits_ind, codebook)
                z_soft_flat = torch.einsum('nlk,nlkd->nld', soft_probs, soft_vecs)
                z_soft = z_soft_flat.permute(0, 2, 1).reshape(-1, self.config.codebook_embed_dim, h, w)
            
            with self.accelerator.autocast():
                fake_policy_soft = unwrapped_vq.decode(z_soft)
            
            target_robust = rec_policy.detach()
            loss_lasc = F.mse_loss(fake_policy_soft.float(), target_robust) * self.config.lambda_lasc_consistency

            del packed_vals, packed_inds, logits_val, logits_ind, soft_vecs, z_soft_flat, z_soft, fake_policy_soft

        # Total Loss & Backward
        loss_policy_total = loss_gan_fake + loss_consist + loss_reward + loss_lasc
        
        if loss_policy_total.requires_grad:
            self.accelerator.backward(loss_policy_total)
        
        gan_item = loss_gan_fake.item()
        consist_item = loss_consist.item()
        reward_item = loss_reward.item()
        lasc_item = loss_lasc.item()
        
        del rec_policy, loss_gan_fake, loss_consist, loss_reward, loss_policy_total, z_q_policy, loss_lasc

        # In Janus mode, all-reduce decoder gradients manually
        if self.config.model_type == 'janus-pro' and self.accelerator.num_processes > 1:
            for param_group in self.decoder_optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
                        param.grad.data.div_(self.accelerator.num_processes)

        if self.config.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(self.vq_model.parameters(), self.config.max_grad_norm)
        
        self.decoder_optimizer.step()
        self.counter_vq_update += 1
        if self.counter_vq_update % self.freqs_ema_vq_update == 0:
            self.update_ema_model(['vq'])
            self.counter_ema_vq_update += 1
            
        metrics.update({
            "loss_dec/gan_recon_gt": gt_loss_item,
            "loss_dec/gan_recon_fake": gan_item,
            "loss_dec/distill_consist": consist_item,
            "loss_dec/reward_loss": reward_item,
            "loss_dec/lasc": lasc_item,
            "loss_dec/decoder_total": gt_loss_item + gan_item + consist_item + reward_item + lasc_item
        })
        
        return metrics

    # =========================================================================
    # Step 4: training-loop orchestrator (LlamaGen-aligned, including consistency scheduling)
    # =========================================================================
    def train_loop_step(self, gt_batch, global_step, total_steps=100000):
        gt_images, gt_prompts = gt_batch
        logs = {}
        
        anneal_steps = int(total_steps * self.config.anneal_ratio)
        eff_prog = min(1.0, global_step / anneal_steps) if anneal_steps > 0 else 1.0
        decay_factor = 0.5 * (1 + math.cos(math.pi * eff_prog))
        
        start_t = self.config.decoder_resample_temp_start
        end_t = self.config.decoder_resample_temp_end
        current_temp = end_t + (start_t - end_t) * decay_factor
        self.config.decoder_resample_temp = max(current_temp, 1e-4)
        logs["param/resample_temp"] = current_temp

        scheduled_consistency = self.get_scheduled_consistency_lambda(global_step)
        self.config.lambda_decoder_consistency = scheduled_consistency
        logs["param/lambda_decoder_consistency"] = scheduled_consistency

        rollout_data = self.generate_rollouts(gt_prompts)
        
        if self.config.train_mode in ['gpt', 'both']:
            gpt_logs, _ = self.update_gpt_policy(rollout_data, global_step)
            logs.update(gpt_logs)
            
        if self.config.train_mode in ['decoder', 'both']:            
            dec_logs = self.update_decoder(rollout_data, gt_images, global_step)
            logs.update(dec_logs)
        
        logs["counter/gpt_opt_step"] = self.counter_gpt_update
        logs["counter/decoder_opt_step"] = self.counter_vq_update
        logs["counter/disc_opt_step"] = self.counter_disc_update
        logs["counter/ema_gpt_update"] = self.counter_ema_gpt_update
        logs["counter/ema_vq_update"] = self.counter_ema_vq_update
            
        return logs

    # =========================================================================
    # Step 5: Sampling Helper
    # =========================================================================
    @torch.no_grad()
    def generate(self, prompts: List[str], max_length: int, use_ema: bool = True):
        gpt_to_use = self.ema_gpt_model if use_ema and self.ema_gpt_model else self.accelerator.unwrap_model(self.gpt_model)
        vq_to_use = self.ema_vq_model if use_ema and self.ema_vq_model else self.accelerator.unwrap_model(self.vq_model)
        
        conds = self.prepare_conditions(prompts, self.accelerator.device)
        
        if self.config.model_type == "llamagen":
            # Simple dtype matching
            try: target_dtype = next(gpt_to_use.parameters()).dtype
            except: target_dtype = torch.bfloat16
            if conds["cond_idx"].dtype != target_dtype: conds["cond_idx"] = conds["cond_idx"].to(target_dtype)
            if conds["mask"].dtype != target_dtype: conds["mask"] = conds["mask"].to(target_dtype)
        
        with self.accelerator.autocast():
            sequences = self.core_generate(gpt_to_use, conds, prompts)
        if not sequences.is_contiguous(): sequences = sequences.contiguous()
        
        B, L = sequences.shape
        h = w = int(math.sqrt(L))
        z_q = vq_to_use.quantize.get_codebook_entry(
            sequences.view(-1), shape=(B, h, w, self.config.codebook_embed_dim), channel_first=False
        ).permute(0, 3, 1, 2)
        
        images = vq_to_use.decode(z_q)
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)
        
        pil_images = []
        for img in images:
            img = img.cpu().float().permute(1, 2, 0).numpy() * 255
            pil_images.append(Image.fromarray(img.astype('uint8')))
        return pil_images
