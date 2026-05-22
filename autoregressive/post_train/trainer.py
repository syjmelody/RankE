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

# Import project modules (make sure PYTHONPATH is set correctly)
from autoregressive.models.generate import generate
from language.t5 import T5Embedder
from autoregressive.models.gpt import Transformer
from transformers import CLIPModel, CLIPProcessor
from tokenizer.tokenizer_image.vq_model import VQModel
from tokenizer.tokenizer_image.vq_loss import VQLoss

from autoregressive.models.vqrl_models import HybridVQLoss, MultiRewardManager

@dataclass
class HybridTrainConfig:
    """
    Hybrid training configuration class
    """
    # --- Strategy ---
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
    grpo_epochs: int = 2
    rollout_length: int = 1024
    clip_eps: float = 0.2
    kl_coef: float = 0.05
    grpo_adv_coef: float = 5.0
    uniform_advantages: bool = False
    
    # --- RankE / Decoder (Stream 2) ---
    lambda_decoder_reward: float = 0.05      # A2: Direct Reward Backprop
    lambda_decoder_gan: float = 0.1          # A1: Weighted GAN
    lambda_lasc_consistency: float = 1.0     # A3: LASC
    lambda_codebook_anchor: float = 10.0     # B2: Anchor
    lambda_reconstruction: float = 1.0       # B1: GT Reconstruction
    lambda_decoder_consistency: float = 5.0  # A3: Consistency Loss
    
    # Decoder Consistency Scheduling
    consistency_schedule_type: str = "none"       # "none" / "linear" / "sin"
    consistency_start_step: int = 0
    consistency_end_step: int = 1000
    consistency_start_value: float = 0.0
    consistency_end_value: float = 5.0
    
    rejection_sample_k: int = 2
    lasc_sample_k: int = 20
    decoder_topk_rewards: int = 1 # Backward-compatible alias
    
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
    gen_adv_loss: str = 'hinge' # Backward-compatible alias
    
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
        # Simple argument mapping for argparse naming differences
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            # Map special abbreviations
            if k == 'lr_gpt': self.learning_rate_gpt = v
            if k == 'lr_decoder': self.learning_rate_decoder = v
            if k == 'lr_disc': self.learning_rate_disc = v
            if k == 'temp_start': self.decoder_resample_temp_start = v
            if k == 'temp_end': self.decoder_resample_temp_end = v



class HybridRLTrainer:
    """
    Optimized hybrid reinforcement-learning trainer
    Integrates GRPO (GPT), RankE (decoder), LASC, multi-reward training, and a DINOv3 discriminator.
    """
    def __init__(
        self,
        gpt_model: Transformer,
        vq_model: VQModel,
        reward_model: MultiRewardManager, # [Change] Accept a MultiRewardManager instance
        t5_model: T5Embedder,
        config: 'HybridTrainConfig',      # Forward type annotation
        accelerator: Accelerator,
        ema_gpt_model: Optional[Transformer] = None,
        ema_vq_model: Optional[VQModel] = None,
        ref_gpt_model: Optional[Transformer] = None,
    ):
        self.gpt_model = gpt_model
        self.vq_model = vq_model
        self.reward_model = reward_model
        self.t5_model = t5_model
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
        # Supports projected DINOv3 GAN training and adaptive weighting
        self.vq_loss = HybridVQLoss(
            disc_start=config.disc_start,
            disc_weight=config.disc_weight,
            disc_type=config.disc_type,
            dino_path=config.dino_path,
            perceptual_weight=1.0,
            reconstruction_weight=config.lambda_reconstruction
        ).to(self.accelerator.device)
        
        # ==================== 3. Freeze static models ====================
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
            self.gpt_optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.gpt_model.parameters()),
                lr=config.learning_rate_gpt, weight_decay=config.weight_decay
            )
        
        if self.config.train_mode in ['decoder', 'both']:
            # Decoder parameters
            dec_params = list(filter(lambda p: p.requires_grad, self.vq_model.parameters()))
            self.decoder_optimizer = optim.AdamW(
                dec_params, lr=config.learning_rate_decoder, weight_decay=config.weight_decay
            )
            # Discriminator parameters (inside vq_loss.discriminator)
            self.disc_optimizer = optim.AdamW(
                self.vq_loss.discriminator.parameters(),
                lr=config.learning_rate_disc, weight_decay=config.weight_decay
            )

        # ==================== 6. Accelerator Prepare ====================
        # Prepare modules manually so object ordering stays explicit
        if self.gpt_optimizer:
            self.gpt_model, self.gpt_optimizer = accelerator.prepare(self.gpt_model, self.gpt_optimizer)
        else:
            self.gpt_model.to(accelerator.device)
            
        if self.decoder_optimizer:
            # Prepare VQ, Optimizer, Disc Optimizer, Loss Module (for sync)
            self.vq_model, self.decoder_optimizer, self.disc_optimizer, self.vq_loss = accelerator.prepare(
                self.vq_model, self.decoder_optimizer, self.disc_optimizer, self.vq_loss
            )
        else:
            self.vq_model.to(accelerator.device)
            self.vq_loss.to(accelerator.device)
        
        # ==================== 7. Counter initialization ====================
        self.counter_gpt_update = 0       # [GPT] actual optimizer.step() count
        self.counter_vq_update = 0        # [Decoder] actual optimizer.step() count
        self.counter_disc_update = 0      # [Disc] actual optimizer.step() count
        self.counter_ema_gpt_update = 0   # [GPT] EMA update count
        self.counter_ema_vq_update = 0    # [Decoder] EMA update count
        
        self.freqs_ema_vq_update: int = config.frqs_ema_vq_update
        self.freqs_ema_gpt_update: int = config.frqs_ema_gpt_update
            
        # CLIP statistics cached as buffers
        self.register_buffer_stats()

    def register_buffer_stats(self):
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.accelerator.device)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.accelerator.device)


    # Utility helpers inside HybridRLTrainer
    def toggle_requires_grad(self, model, requires_grad: bool):
        """Safely enable or disable gradients for all parameters in a model."""
        for param in model.parameters():
            param.requires_grad = requires_grad


    # ==================== Helper Functions ====================
    def clear_kv_cache(self, model):
        """Explicitly clear KV cache tensors to reduce OOM risk."""
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
        """
        Compute lambda_decoder_consistency dynamically from schedule_type.
        Supports both linear and sin (cosine annealing) schedules.
        When schedule_type == "none", return the static config value directly.
        """
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
            # cosine annealing: start_val -> end_val
            return end_val + (start_val - end_val) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            return cfg.lambda_decoder_consistency

    def update_ema_model(self, type: List[str] = ['gpt', 'vq']):
        with torch.no_grad():
            if 'gpt' in type:
                current_gpt = self.accelerator.unwrap_model(self.gpt_model)
                decay = self.config.ema_decay_gpt
                for param_q, param_k in zip(current_gpt.parameters(), self.ema_gpt_model.parameters()):
                    param_k.data.mul_(decay).add_(param_q.data, alpha=1 - decay)
            if 'vq' in type:
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
        if self.config.uniform_advantages:
            return torch.ones_like(rewards)

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

    
    def get_per_token_logps(self, model, text_embs, sequences, emb_masks):
        """
        [Optimization] Includes the slice fix so logits stay aligned with image tokens
        """
        was_training = model.training
        model.eval()
        
        # [Fix 1] Unwrap DDP so model-specific attributes remain accessible
        unwrapped_model = self.accelerator.unwrap_model(model)

        self.clear_kv_cache(model)

        if hasattr(model, "gradient_checkpointing_enable") and self.config.gradient_checkpointing:
             model.gradient_checkpointing_enable()

        B, L_img = sequences.shape
        T_text = text_embs.shape[1]
        device = sequences.device

        input_ids = sequences[:, :-1]
        targets = sequences 

        # Mask Construction
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

        # Forward
        # [Fix 3] Some LlamaGen forward implementations expose use_cache; set it to False when available
        # This path assumes the argument is absent and relies on clearing kv_cache attributes instead
        logits, _ = model(idx=input_ids.long(), cond_idx=text_embs, input_pos=input_pos, mask=full_mask)
        
        image_logits = logits[:, T_text - 1 :, :]
        
        logits_flat = image_logits.reshape(-1, image_logits.size(-1))
        targets_flat = targets.reshape(-1).long()
        nll_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        token_lp = -nll_loss.view(B, targets.shape[1])

        self.clear_kv_cache(model)

        if was_training: model.train()
        return token_lp, image_logits
    

    # ==================== Step 1: Rollout & Rewards ====================
    @torch.no_grad()
    def generate_rollouts(self, batch_prompts: List[str]):
        """
        Generate rollouts, compute multi-reward scores, and prepare logits for LASC.
        """
        self.gpt_model.eval()
        
        # 1. Text Embedding
        caption_embs, emb_masks = self.t5_model.get_text_embeddings(batch_prompts)
        
        # 2. Expand for Group Size (GRPO)
        B, G = len(batch_prompts), self.config.group_size
        c_indices = caption_embs.repeat_interleave(G, dim=0)
        c_masks = emb_masks.repeat_interleave(G, dim=0)
        
        # 3. Generate Sequences
        with self.accelerator.autocast():
            sequences = generate(
                self.accelerator.unwrap_model(self.gpt_model),
                c_indices, self.config.rollout_length, emb_masks=c_masks,
                cfg_scale=self.config.gen_cfg_scale, temperature=self.config.gen_temperature,
                top_k=self.config.gen_top_k, sample_logits=True
            )
        if not sequences.is_contiguous(): sequences = sequences.contiguous()
        
        self.clear_kv_cache(self.gpt_model)      # Clear the main model cache
        self.clear_kv_cache(self.ema_gpt_model)  # [Fix] Explicitly clear the EMA model cache
        
        # 4. Compute Log Probs
        with self.accelerator.autocast():
            old_log_probs, full_logits = self.get_per_token_logps(self.gpt_model, c_indices, sequences, c_masks)
            
            # [Change 2] Prefer ref_gpt_model, otherwise fall back to ema_gpt_model
            ref_model_to_use = self.ref_gpt_model if self.ref_gpt_model is not None else self.ema_gpt_model
            if ref_model_to_use is not None:
                ref_log_probs, _ = self.get_per_token_logps(ref_model_to_use, c_indices, sequences, c_masks)
            else:
                # If neither exists, reuse old_log_probs so KL becomes zero
                ref_log_probs = old_log_probs.clone()
        
        # 5. Compute Rewards
        # As in the earlier pipeline, decode with EMA VQ so reward signals stay stable while the decoder changes
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
        
        # Query MultiRewardManager
        expanded_prompts = [p for p in batch_prompts for _ in range(G)]
        # Returns (total weighted reward, breakdown dict)
        final_rewards, reward_breakdown = self.reward_model(images, expanded_prompts)
        
        # 6. Store Top-K Logits for LASC (CPU Offload to save VRAM)
        K_lasc = self.config.lasc_sample_k
        top_k_vals, top_k_inds = torch.topk(full_logits, K_lasc, dim=-1)
        packed_logits = {
            "values": top_k_vals.detach().cpu(),
            "indices": top_k_inds.detach().cpu()
        }
        
        if self.config.train_mode in ['gpt', 'both']:
            self.gpt_model.train()
            
        return {
            "prompts": batch_prompts, "text_embs": c_indices, "emb_masks": c_masks,
            "sequences": sequences.detach(), "old_log_probs": old_log_probs.detach(),
            "ref_log_probs": ref_log_probs.detach(),
            "rewards": final_rewards.to(self.accelerator.device).detach(),
            "reward_breakdown": {k: v.detach() if torch.is_tensor(v) else v for k, v in reward_breakdown.items()},
            "packed_logits": packed_logits,
            "group_ids": torch.arange(B, device=self.accelerator.device).repeat_interleave(G)
        }
    

    # ==================== Step 2: GPT Policy Optimization ====================
    def update_gpt_policy(self, data: Dict, glob_step: int):
        # Unpack pre-calculated values
        sequences = data["sequences"]
        text_embs = data["text_embs"]
        emb_masks = data["emb_masks"]
        old_log_probs = data["old_log_probs"].detach() 
        ref_log_probs = data["ref_log_probs"].to(self.accelerator.device).detach()
        rewards = data["rewards"] # Uses pre-calculated rewards
        reward_breakdown = data.get("reward_breakdown", {})
        group_ids = data["group_ids"]

        advantages = self.compute_grpo_advantages(rewards, group_ids).detach()
        advantages = advantages

        self.gpt_model.train()
        if hasattr(self.gpt_model, "gradient_checkpointing_enable") and self.config.gradient_checkpointing:
             self.gpt_model.gradient_checkpointing_enable()

        metrics_accum = defaultdict(list)
        valid_mask = torch.ones_like(sequences, dtype=torch.float32, device=self.accelerator.device)

        for i_epoch in range(self.config.grpo_epochs):
            self.gpt_optimizer.zero_grad() 
            new_log_probs, _ = self.get_per_token_logps(self.gpt_model, text_embs, sequences, emb_masks)

            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            
            adv_expanded = advantages.unsqueeze(1).expand_as(ratio)
            surr1 = ratio * adv_expanded
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * adv_expanded
            policy_loss = -torch.min(surr1, surr2).mean()

            kl_div = 0.5 * (new_log_probs - ref_log_probs)**2
            kl_loss = (kl_div * valid_mask).sum() / (valid_mask.sum() + 1e-8)

            total_loss = policy_loss + self.config.kl_coef * kl_loss
            
            # Update gradients
            self.accelerator.backward(total_loss)
            if self.accelerator.sync_gradients:
                if self.config.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.gpt_model.parameters(), self.config.max_grad_norm)
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
    
    
    # ==================== Step 3: Decoder Update (RankE + LASC + Precision Fix) ====================
    def update_decoder(self, rollout_data: Dict, gt_images: torch.Tensor, global_step: int):
        """
        Optimized decoder-update logic.
        Core strategy:
        1. RankE: compute sample weights weights_flat from reward values.
        2. FP32 anchor: cast decoder outputs to float32 immediately for stable downstream losses.
        3. Autocast: run decoder forward passes in bf16 to save memory.
        """
        self.vq_model.train()
        
        unwrapped_vq = self.accelerator.unwrap_model(self.vq_model)
        unwrapped_vq_loss = self.accelerator.unwrap_model(self.vq_loss)
        
        metrics = {}
        device = self.accelerator.device
        
        # Gradient Checkpointing
        if self.config.gradient_checkpointing:
            if hasattr(unwrapped_vq.decoder, "gradient_checkpointing_enable"):
                unwrapped_vq.decoder.gradient_checkpointing_enable()
            elif hasattr(unwrapped_vq.decoder, "gradient_checkpointing"): 
                unwrapped_vq.decoder.gradient_checkpointing = True

        # --- RankE Weights Calculation ---
        rewards = rollout_data["rewards"]
        B, G = self.config.batch_size, self.config.group_size
        K = min(self.config.rejection_sample_k, G)
        
        rewards_reshaped = rewards.view(B, G)
        sorted_vals, sorted_indices = torch.sort(rewards_reshaped, dim=1, descending=True)
        top_k_indices = sorted_indices[:, :K]
        top_k_rewards = sorted_vals[:, :K]
        
        # Softmax weighting based on rewards
        weights = F.softmax(top_k_rewards / self.config.decoder_resample_temp, dim=1).detach()
        weights_flat = weights.view(-1)
        weights_flat = weights_flat / (weights_flat.mean() + 1e-8) # Normalize mean to 1
        
        # ---------------------------------------------------------
        # 1. Data preparation (no grad)
        # ---------------------------------------------------------
        with torch.no_grad():
            # A. GT Encoding (FP32 Anchor)
            with torch.cuda.amp.autocast(enabled=False):
                z_q_gt, _, _ = unwrapped_vq.encode(gt_images.float())
            
            # B. Gather Top-K Sequences
            sequences = rollout_data["sequences"].detach().reshape(B, G, -1)
            L_seq = sequences.shape[-1]
            expanded_indices = top_k_indices.unsqueeze(-1).expand(B, K, L_seq)
            selected_sequences = torch.gather(sequences, 1, expanded_indices).reshape(-1, L_seq)
            
            # Codebook Lookup (FP32)
            B_sel = selected_sequences.shape[0]
            h = w = int(math.sqrt(L_seq))
            with torch.cuda.amp.autocast(enabled=False):
                z_q_policy = unwrapped_vq.quantize.get_codebook_entry(
                    selected_sequences.view(-1), shape=(B_sel, h, w, self.config.codebook_embed_dim), channel_first=False
                ).permute(0, 3, 1, 2).float()

            batch_prompts = rollout_data["prompts"]
            target_prompts = [p for p in batch_prompts for _ in range(K)]
            
            del sequences, expanded_indices, selected_sequences

        # =====================================================================
        # Stage 1: update the discriminator
        # =====================================================================
        # 1. Unfreeze the discriminator so it can receive gradients
        self.toggle_requires_grad(unwrapped_vq_loss.discriminator, True)
        self.disc_optimizer.zero_grad()

        with torch.no_grad():
            with self.accelerator.autocast():
                # Must detach here to prevent gradients from flowing back into the generator
                rec_gt_detached = unwrapped_vq.decode(z_q_gt).detach()

        # Use the DDP-wrapped vq_loss module to compute and synchronize discriminator loss
        loss_disc, _ = self.vq_loss(
            inputs=gt_images.float(), 
            reconstructions=rec_gt_detached.float(), 
            optimizer_idx=1, global_step=global_step, last_layer=None
        )
        self.accelerator.backward(loss_disc)
        self.disc_optimizer.step()
        self.counter_disc_update += 1
        metrics["loss_dec/disc_loss"] = loss_disc.item()


        # =====================================================================
        # Stage 2: update the generator/decoder
        # =====================================================================
        # 2. Freeze the discriminator completely to avoid DDP "ready twice" errors
        self.toggle_requires_grad(unwrapped_vq_loss.discriminator, False)
        
        self.decoder_optimizer.zero_grad()
        last_layer = unwrapped_vq.decoder.last_layer if hasattr(unwrapped_vq.decoder, "last_layer") else None

        # --- Stream A: GT Recon ---
        gt_loss_item = 0.0
        if self.config.lambda_reconstruction > 0:
            with self.accelerator.autocast():
                rec_gt = unwrapped_vq.decode(z_q_gt)
            
            # Reuse the DDP-wrapped vq_loss module, where optimizer_idx=0 selects generator reconstruction loss
            loss_gen_gt, _ = self.vq_loss(
                inputs=gt_images.float(), reconstructions=rec_gt.float(),
                optimizer_idx=0, global_step=global_step, last_layer=last_layer
            )
            loss_gen_gt = loss_gen_gt * self.config.lambda_reconstruction
            self.accelerator.backward(loss_gen_gt)
            gt_loss_item = loss_gen_gt.item()

        # --- Stream B: Policy Optimization (RankE Weighted GAN & Rewards) ---
        with self.accelerator.autocast():
            # Generate fake images while keeping gradients
            rec_policy = unwrapped_vq.decode(z_q_policy.requires_grad_(True))
        rec_policy = torch.clamp(rec_policy, -1.0, 1.0).float()

        loss_gan_fake = torch.tensor(0., device=device)
        if global_step >= self.config.disc_start:
            with self.accelerator.autocast():
                # Critical: use unwrapped_vq_loss.discriminator here
                # Do not let the rec_policy graph flow through the DDP shell self.vq_loss
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
            
            # Inputs are all float tensors
            p_loss = unwrapped_vq_loss.perceptual_loss(rec_policy, ema_targets.float()).mean()
            loss_consist = p_loss * self.config.lambda_decoder_consistency
            del ema_targets

        # 4. Direct Reward Loss (RankE Weighted)
        loss_reward = torch.tensor(0., device=device)
        if self.config.lambda_decoder_reward > 0:
            rec_norm = torch.clamp((rec_policy + 1.0) * 0.5, 0.0, 1.0) # Float
            rewards_diff, _ = self.reward_model(rec_norm, target_prompts)
            loss_reward = -(rewards_diff * weights_flat).mean() * self.config.lambda_decoder_reward

        # >>> Stream C: LASC <<<
        loss_lasc = torch.tensor(0.0, device=device)
        if self.config.lambda_lasc_consistency > 0:
            # 1. Recover Logits
            packed_vals = rollout_data["packed_logits"]["values"].to(device, non_blocking=True)
            packed_inds = rollout_data["packed_logits"]["indices"].to(device, non_blocking=True)
            K_sample = packed_vals.shape[-1]
            logits_val = packed_vals.view(B, G, L_seq, K_sample)
            logits_ind = packed_inds.view(B, G, L_seq, K_sample)
            
            # 2. Gather Top-K
            gather_idx = top_k_indices.view(B, K, 1, 1).expand(-1, -1, L_seq, K_sample)
            sel_logits_val = torch.gather(logits_val, 1, gather_idx).reshape(-1, L_seq, K_sample)
            sel_logits_ind = torch.gather(logits_ind, 1, gather_idx).reshape(-1, L_seq, K_sample)
            
            # 3. Soft Summation (FP32)
            with torch.cuda.amp.autocast(enabled=False):
                soft_probs = F.softmax(sel_logits_val.float(), dim=-1)
                codebook = unwrapped_vq.quantize.embedding.weight.float() 
                soft_vecs = F.embedding(sel_logits_ind, codebook)
                z_soft_flat = torch.einsum('nlk,nlkd->nld', soft_probs, soft_vecs)
                z_soft = z_soft_flat.permute(0, 2, 1).reshape(-1, self.config.codebook_embed_dim, h, w)
            
            # 4. Decode Soft (Autocast)
            with self.accelerator.autocast():
                fake_policy_soft = unwrapped_vq.decode(z_soft)
            
            # 5. Loss (Float32)
            target_robust = rec_policy.detach() # Already Float
            loss_lasc = F.mse_loss(fake_policy_soft.float(), target_robust) * self.config.lambda_lasc_consistency

            del packed_vals, packed_inds, logits_val, logits_ind, soft_vecs, z_soft_flat, z_soft, fake_policy_soft

        # Total Loss & Backward
        loss_policy_total = loss_gan_fake + loss_consist + loss_reward + loss_lasc
        
        if loss_policy_total.requires_grad:
            self.accelerator.backward(loss_policy_total)
        
        # Logging
        gan_item = loss_gan_fake.item()
        consist_item = loss_consist.item()
        reward_item = loss_reward.item()
        lasc_item = loss_lasc.item()
        
        del rec_policy, loss_gan_fake, loss_consist, loss_reward, loss_policy_total, z_q_policy, loss_lasc

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
    

    # ==================== Step 4: Train Loop Orchestrator ====================
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
    
    
    # ==================== Step 5: Sampling Helper ====================
    @torch.no_grad()
    def generate(self, prompts: List[str], max_length: int, use_ema: bool = True):
        # 1. Select Models
        gpt_to_use = self.ema_gpt_model if use_ema else self.accelerator.unwrap_model(self.gpt_model)
        vq_to_use = self.ema_vq_model if use_ema else self.accelerator.unwrap_model(self.vq_model)
        
        # 2. Encode
        caption_embs, emb_masks = self.t5_model.get_text_embeddings(prompts)
        # Simple dtype matching
        try: target_dtype = next(gpt_to_use.parameters()).dtype
        except: target_dtype = torch.bfloat16
        if caption_embs.dtype != target_dtype: caption_embs = caption_embs.to(target_dtype)
        if emb_masks.dtype != target_dtype: emb_masks = emb_masks.to(target_dtype)
        
        # Simple padding logic (keeps the implementation concise by omitting reversal)
        
        # 3. Generate
        with self.accelerator.autocast():
            sequences = generate(
                gpt_to_use, caption_embs, max_length, emb_masks=emb_masks,
                cfg_scale=self.config.gen_cfg_scale, temperature=self.config.gen_temperature,
                top_k=self.config.gen_top_k, sample_logits=True
            )
        if not sequences.is_contiguous(): sequences = sequences.contiguous()
        
        # 4. Decode
        B, L = sequences.shape
        h = w = int(math.sqrt(L))
        z_q = vq_to_use.quantize.get_codebook_entry(
            sequences.view(-1), shape=(B, h, w, self.config.codebook_embed_dim), channel_first=False
        ).permute(0, 3, 1, 2)
        
        images = vq_to_use.decode(z_q)
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)
        
        # 5. PIL
        pil_images = []
        for img in images:
            img = img.cpu().float().permute(1, 2, 0).numpy() * 255
            pil_images.append(Image.fromarray(img.astype('uint8')))
        return pil_images