import argparse
import math
import os
import sys
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
import wandb
from torchvision import transforms
from tqdm import tqdm

# =============================================================================
# Dependency declaration (PYTHONPATH is set by the shell script)
# =============================================================================
try:
    from dataset.sft_dataset import build_dataloader
    from autoregressive.models.vqrl_models import MultiRewardManager
    from janus.post_train.trainer import UnifiedHybridRLTrainer, HybridTrainConfig
except ImportError as e:
    print(f">>> [Error] Failed to import core modules. Please check that PYTHONPATH is configured correctly. Details: {e}")
    sys.exit(1)

# Global GPU acceleration settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import copy

def load_unified_models(args, device):
    """
    Unified model-loading factory (memory-optimized version)
    """
    torch.cuda.set_device(device)
    
    print(f">>> [Loader] Initializing Unified Framework for: {args.model_type.upper()} on {device}")
    
    processor = None
    t5_model = None
    ref_gpt_model = None
    
    if args.model_type == "llamagen":
        import autoregressive.models.gpt as GPT_models_module
        GPT_models = GPT_models_module.GPT_models
        from tokenizer.tokenizer_image.vq_model import VQ_models
        from language.t5 import T5Embedder
        
        # 1. T5
        print(f">>> [Loader] Loading T5 ({args.t5_model_type})")
        t5_model = T5Embedder(
            device=device, local_cache=True, cache_dir=args.t5_path, 
            dir_or_name=args.t5_model_type,
            torch_dtype=torch.bfloat16 if args.mixed_precision == 'bf16' else torch.float32, 
            model_max_length=args.t5_max_len
        )

        # 2. VQ
        print(f">>> [Loader] Loading LlamaGen VQ ({args.vq_model})")
        vq_model = VQ_models[args.vq_model](codebook_size=args.codebook_size, codebook_embed_dim=args.codebook_embed_dim)
        vq_ckpt = torch.load(args.vq_ckpt, map_location='cpu')
        vq_state = vq_ckpt['model'] if isinstance(vq_ckpt, dict) and 'model' in vq_ckpt else vq_ckpt
        vq_model.load_state_dict(vq_state)
        vq_model.to(device).eval()
        
        ema_vq_model = copy.deepcopy(vq_model).eval().requires_grad_(False)
        
        # 3. GPT
        print(f">>> [Loader] Loading LlamaGen GPT from: {args.gpt_ckpt}")
        latent_size = args.image_size // args.downsample_size
        gpt_model = GPT_models[args.gpt_model](block_size=latent_size**2, cls_token_num=args.cls_token_num, model_type='t2i')
        gpt_ckpt = torch.load(args.gpt_ckpt, map_location='cpu')
        
        state_dict = gpt_ckpt['model'] if isinstance(gpt_ckpt, dict) and 'model' in gpt_ckpt else (
                     gpt_ckpt['state_dict'] if isinstance(gpt_ckpt, dict) and 'state_dict' in gpt_ckpt else gpt_ckpt)
                     
        new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        gpt_model.load_state_dict(new_state_dict, strict=False)
        gpt_model.to(device)
        
        print(f">>> [Loader] Cloning LlamaGen EMA & Ref Models in VRAM...")
        ema_gpt_model = copy.deepcopy(gpt_model).eval().requires_grad_(False)

        if args.kl_coef > 0 and args.use_fixed_ref_model:
            ref_gpt_model = copy.deepcopy(gpt_model).eval().requires_grad_(False)
        
    elif args.model_type == "janus-pro":
        from janus.models import MultiModalityCausalLM, VLChatProcessor
        
        print(f">>> [Loader] Loading Janus-Pro Processor")
        processor = VLChatProcessor.from_pretrained(args.gpt_ckpt)
        
        orig_linspace = torch.linspace
        torch.linspace = lambda *a, **k: orig_linspace(*a, **{**k, 'device': 'cpu'})
        
        print(f">>> [Loader] Loading Janus-Pro Main Model (Single Load from Disk)")
        gpt_model = MultiModalityCausalLM.from_pretrained(
            args.gpt_ckpt, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=False, 
            _fast_init=False
        ).to(device)
        torch.linspace = orig_linspace
        
        print(f">>> [Loader] Cloning Janus-Pro EMA & Ref Models in VRAM...")
        ema_gpt_model = copy.deepcopy(gpt_model).eval().requires_grad_(False)
        
        if args.kl_coef > 0 and args.use_fixed_ref_model:
            ref_gpt_model = copy.deepcopy(gpt_model).eval().requires_grad_(False)
            
        vq_model = gpt_model.gen_vision_model
        ema_vq_model = ema_gpt_model.gen_vision_model

    # Enable gradient checkpointing consistently
    if hasattr(gpt_model, "language_model") and hasattr(gpt_model.language_model, "gradient_checkpointing_enable"):
        gpt_model.language_model.gradient_checkpointing_enable()
    elif hasattr(gpt_model, "gradient_checkpointing_enable"):
        gpt_model.gradient_checkpointing_enable()

    torch.cuda.empty_cache()

    # Reward Manager
    print(">>> [Loader] Initializing MultiReward Manager")
    class RewardConfigStub:
        def __init__(self):
            self.reward_weights = {
                "clip_score": args.reward_weight_clip,
                "aesthetic_score": args.reward_weight_aesthetic,
                "image_reward": args.reward_weight_image_reward,
                "hpsv2": args.reward_weight_hpsv2
            }
            self.reward_paths = {
                "clip_score": args.reward_path_clip,
                "aesthetic_score": args.reward_path_aesthetic,
                "image_reward": args.reward_path_image_reward,
                "hpsv2": args.reward_path_hpsv2
            }
            self.normalize_rewards = False
    reward_model = MultiRewardManager(RewardConfigStub(), device)
    
    return gpt_model, vq_model, t5_model, processor, reward_model, ref_gpt_model, ema_gpt_model, ema_vq_model

def main():
    parser = argparse.ArgumentParser(description="Unified hybrid RL training entry point")
    
    # Primary framework switch
    parser.add_argument("--model-type", type=str, choices=["llamagen", "janus-pro"], required=True)
    
    # --- Data & Paths ---
    parser.add_argument("--dataset-name", type=str, default="scaling_simple")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # --- Infrastructure ---
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--t5-path", type=str, default="./pretrained/t5")
    parser.add_argument("--resume-from", type=str, default=None)
    
    # --- Rewards ---
    parser.add_argument("--reward-path-clip", type=str, default=None)
    parser.add_argument("--reward-path-aesthetic", type=str, default=None)
    parser.add_argument("--reward-path-image-reward", type=str, default=None)
    parser.add_argument("--reward-path-hpsv2", type=str, default=os.getenv("HPS_ROOT"))

    # --- Training ---
    parser.add_argument("--train-mode", type=str, default="both", choices=["gpt", "decoder", "both"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--mixed-precision", type=str, default="bf16")
    parser.add_argument("--sync-gpt-decoder-update", action="store_true")

    # --- Optimizer ---
    parser.add_argument("--lr-gpt", type=float, default=1e-6)
    parser.add_argument("--lr-decoder", type=float, default=1e-5)
    parser.add_argument("--lr-disc", type=float, default=1e-5)
    parser.add_argument("--lr-codebook", type=float, default=0.0)

    # --- GRPO ---
    parser.add_argument("--grpo-epochs", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--grpo-adv-coef", type=float, default=1.0)
    parser.add_argument("--rollout-length", type=int, default=256)

    # --- Decoder Loss ---
    parser.add_argument("--lambda-decoder-reward", type=float, default=0.05)
    parser.add_argument("--lambda-decoder-gan", type=float, default=2.0)
    parser.add_argument("--lambda-decoder-consistency", type=float, default=0.0)
    parser.add_argument("--lambda-reconstruction", type=float, default=1.0)
    parser.add_argument("--lambda-codebook-anchor", type=float, default=10.0)
    parser.add_argument("--lambda-lasc-consistency", type=float, default=0.0)

    # --- Decoder Consistency Scheduling ---
    parser.add_argument("--consistency-schedule-type", type=str, default="none", choices=["none", "linear", "sin"])
    parser.add_argument("--consistency-start-step", type=int, default=0)
    parser.add_argument("--consistency-end-step", type=int, default=1000)
    parser.add_argument("--consistency-start-value", type=float, default=0.0)
    parser.add_argument("--consistency-end-value", type=float, default=5.0)

    # --- Sampling ---
    parser.add_argument("--rejection-sample-k", type=int, default=1)
    parser.add_argument("--lasc-sample-k", type=int, default=20)
    parser.add_argument("--temp-start", type=float, default=4.0)
    parser.add_argument("--temp-end", type=float, default=1.0)
    parser.add_argument("--anneal-ratio", type=float, default=0.2)
    parser.add_argument("--decoder-resample-temp", type=float, default=1.0)

    # --- Disc ---
    parser.add_argument("--disc-start", type=int, default=100)
    parser.add_argument("--disc-weight", type=float, default=0.5)
    parser.add_argument("--disc-type", type=str, default="patchgan")
    parser.add_argument("--dino-path", type=str, default=None)
    parser.add_argument("--reward-weight-clip", type=float, default=1.0)
    parser.add_argument("--reward-weight-aesthetic", type=float, default=0.0)
    parser.add_argument("--reward-weight-image-reward", type=float, default=0.0)
    parser.add_argument("--reward-weight-hpsv2", type=float, default=0.0)

    # --- Model Specs ---
    parser.add_argument("--gen-cfg-scale", type=float, default=1.0)
    parser.add_argument("--gen-top-k", type=int, default=1000)
    parser.add_argument("--gen-temperature", type=float, default=1.0)
    parser.add_argument("--ema-decay-vq", type=float, default=0.99)
    parser.add_argument("--ema-decay-gpt", type=float, default=0.999)
    parser.add_argument("--frqs-ema-gpt-update", type=int, default=10)
    parser.add_argument("--frqs-ema-vq-update", type=int, default=1)
    parser.add_argument("--use-fixed-ref-model", action="store_true")

    parser.add_argument("--image-size", type=int, default=256)
    
    # LlamaGen-specific specs (ignored by Janus)
    parser.add_argument("--gpt-model", type=str, default="GPT-XL")
    parser.add_argument("--vq-model", type=str, default="VQ-16")
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--cls-token-num", type=int, default=120)
    parser.add_argument("--t5-model-type", type=str, default="flan-t5-xl")
    parser.add_argument("--t5-max-len", type=int, default=120)

    # --- Logging ---
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="ranke_janus")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=250)

    args = parser.parse_args()

    # ==================== Setup Accelerator ====================
    needs_find_unused = True if args.model_type == 'janus-pro' else False
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=needs_find_unused)
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=project_config,
        log_with="wandb" if args.use_wandb else None,
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process and args.use_wandb:
        accelerator.init_trackers(
            args.wandb_project, 
            config=vars(args),
            init_kwargs={"wandb": {"name": args.wandb_run_name, "entity": args.wandb_entity}} if args.wandb_run_name else {}
        )
    
    set_seed(args.seed)

    # ==================== Data Loading ====================
    train_transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataloader, _, steps_per_epoch = build_dataloader(args, train_transform, accelerator.num_processes, accelerator.process_index)
    
    # ==================== Models & Trainer ====================
    (gpt_model, vq_model, t5_model, processor, reward_model, 
     ref_gpt_model, ema_gpt_model, ema_vq_model) = load_unified_models(args, accelerator.device)

    reward_weights = {
        "clip_score": args.reward_weight_clip,
        "aesthetic_score": args.reward_weight_aesthetic,
        "image_reward": args.reward_weight_image_reward,
        "hpsv2": args.reward_weight_hpsv2
    }
    
    valid_config_keys = set(HybridTrainConfig.__annotations__.keys())
    config_dict = {k: v for k, v in vars(args).items() if k in valid_config_keys}
    config_dict['reward_weights'] = reward_weights
    
    train_config = HybridTrainConfig(**config_dict)
    train_config.model_type = args.model_type

    trainer = UnifiedHybridRLTrainer(
        gpt_model=gpt_model, 
        vq_model=vq_model, 
        reward_model=reward_model, 
        t5_model=t5_model,
        processor=processor,
        config=train_config, 
        accelerator=accelerator,
        ref_gpt_model=ref_gpt_model, 
        ema_gpt_model=ema_gpt_model, 
        ema_vq_model=ema_vq_model,
    )

    # ==================== Training loop (aligned with the LlamaGen pipeline) ====================
    global_step = 0
    start_epoch = 0
    
    if args.resume_from and os.path.exists(args.resume_from):
        accelerator.load_state(args.resume_from)
        try:
            step_str = os.path.basename(args.resume_from.rstrip('/')).split('_')[-1]
            global_step = int(step_str)
            accelerator.print(f"Resumed from step {global_step}")
        except: pass

    if steps_per_epoch > 0:
        total_steps = steps_per_epoch * args.max_epochs
    else:
        total_steps = 100000 
        
    progress_bar = tqdm(range(global_step, total_steps), disable=not accelerator.is_local_main_process)
    
    accelerator.print(f"Start Unified Training [{args.model_type.upper()}] | Mode: {args.train_mode.upper()} | Total Steps: {total_steps}")
            
    for epoch in range(start_epoch, args.max_epochs):
        for batch in dataloader:
            gt_images, gt_prompts = batch
            gt_images = gt_images.to(accelerator.device, non_blocking=True)
            batch_data = (gt_images, gt_prompts)

            # ----------------------------------------------
            # [a]. Sync GPT & Decoder Update
            # ----------------------------------------------
            if args.sync_gpt_decoder_update:
                models_to_sync = []
                if args.train_mode in ['gpt', 'both']:
                    models_to_sync.append(trainer.gpt_model)
                if args.train_mode in ['decoder', 'both']:
                    if args.model_type == 'llamagen':
                        models_to_sync.append(trainer.vq_model)
                        models_to_sync.append(trainer.vq_loss)

                with accelerator.accumulate(*models_to_sync):
                    logs = trainer.train_loop_step(batch_data, global_step, total_steps=total_steps)

            # ----------------------------------------------
            # [b] Non-sync mode: follow the inline LlamaGen update flow
            # Accumulate GPT gradients while updating the decoder every step
            # ----------------------------------------------
            else:
                gt_images, gt_prompts = batch_data
                logs = {}

                # Annealing
                anneal_steps = int(total_steps * trainer.config.anneal_ratio)
                eff_prog = min(1.0, global_step / anneal_steps) if anneal_steps > 0 else 1.0
                decay_factor = 0.5 * (1 + math.cos(math.pi * eff_prog))
                start_t = trainer.config.decoder_resample_temp_start
                end_t = trainer.config.decoder_resample_temp_end
                current_temp = end_t + (start_t - end_t) * decay_factor
                trainer.config.decoder_resample_temp = max(current_temp, 1e-4)
                logs["param/resample_temp"] = current_temp

                scheduled_consistency = trainer.get_scheduled_consistency_lambda(global_step)
                trainer.config.lambda_decoder_consistency = scheduled_consistency
                logs["param/lambda_decoder_consistency"] = scheduled_consistency

                # 1. Shared rollout (no grad)
                with torch.no_grad():
                    rollout_data = trainer.generate_rollouts(gt_prompts)

                current_step_reward = rollout_data["rewards"].mean()
                global_reward = accelerator.reduce(current_step_reward, reduction="mean").item()
                logs["reward/clip_mean"] = global_reward

                # ==========================================
                # Strategy A: GPT policy update with gradient accumulation
                # ==========================================
                if args.train_mode in ['gpt', 'both']:
                    with accelerator.accumulate(trainer.gpt_model):
                        gpt_logs, _ = trainer.update_gpt_policy(rollout_data, global_step)
                        logs["lr/gpt"] = trainer._get_current_lr(trainer.gpt_optimizer)
                    logs.update(gpt_logs)
                    del gpt_logs
                    torch.cuda.empty_cache()

                # ==========================================
                # Strategy B: VQ decoder fast update at every step
                # ==========================================
                if args.train_mode in ['decoder', 'both']:
                    dec_logs = trainer.update_decoder(rollout_data, gt_images, global_step)
                    logs["lr/decoder"] = trainer._get_current_lr(trainer.decoder_optimizer)
                    logs["lr/disc"] = trainer._get_current_lr(trainer.disc_optimizer)
                    logs.update(dec_logs)
                    del dec_logs
                    torch.cuda.empty_cache()

            # ==========================================
            # Global step bookkeeping (kept aligned with LlamaGen)
            # ==========================================
            global_step += 1
            # --- Logging ---
            if accelerator.is_main_process:
                if global_step % args.log_interval == 0:
                    rew_val = logs.get('reward/clip_mean', logs.get('reward/total_mean', 0))
                    postfix_logs = {"rew": f"{rew_val:.2f}"}

                    if 'reward/raw_clip_score' in logs:
                        postfix_logs["clip"] = f"{logs['reward/raw_clip_score']:.3f}"
                    if 'reward/raw_aesthetic_score' in logs:
                        postfix_logs["aes"] = f"{logs['reward/raw_aesthetic_score']:.3f}"
                    if 'reward/raw_hpsv2' in logs:
                        postfix_logs["hpsv2"] = f"{logs['reward/raw_hpsv2']:.3f}"
                    if 'reward/total_mean' in logs:
                        postfix_logs["rew"] = f"{logs['reward/total_mean']:.2f}"

                    if 'loss_dec/gan_recon_fake' in logs:
                        postfix_logs["gan"] = f"{logs['loss_dec/gan_recon_fake']:.2f}"
                    if 'loss_gpt/policy_loss' in logs:
                        postfix_logs["g_ls"] = f"{logs['loss_gpt/policy_loss']:.4f}"

                    if 'lr/gpt' in logs:
                        postfix_logs["lr_g"] = f"{logs['lr/gpt']:.1e}"
                    if 'lr/decoder' in logs:
                        postfix_logs["lr_d"] = f"{logs['lr/decoder']:.1e}"

                    progress_bar.set_postfix(postfix_logs, refresh=False)
                    accelerator.log(logs, step=global_step)
                progress_bar.update(1)

            if global_step % 100 == 0:
                accelerator.print("-" * 100)
                accelerator.print(f"Global Step: {global_step}")
                accelerator.print(f"Model Type: {args.model_type} | Sync GPT & Decoder Update: {args.sync_gpt_decoder_update}")
                accelerator.print(f"Dynamic Update Counter: GPT: {trainer.counter_gpt_update}, VQ: {trainer.counter_vq_update}")
                accelerator.print(f"EMA Update Counter: GPT: {trainer.counter_ema_gpt_update}, VQ: {trainer.counter_ema_vq_update}")
                accelerator.print(f"EMA Update Frequency: GPT: {trainer.freqs_ema_gpt_update}, VQ: {trainer.freqs_ema_vq_update}")
                accelerator.print("-" * 100)

            is_save_step = (global_step % args.save_interval == 0)
            is_sample_step = (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0))

            if is_save_step or is_sample_step:
                accelerator.wait_for_everyone()
                
                # A. Save Checkpoint
                if is_save_step:
                    save_path = os.path.join(args.output_dir, f"checkpoint_{global_step}")
                    accelerator.save_state(save_path)
                    
                    if accelerator.is_main_process:
                        os.makedirs(save_path, exist_ok=True)
                        torch.save(args, os.path.join(save_path, "training_args.pt"))
                        
                        if args.train_mode in ['gpt', 'both']:
                            unwrapped_gpt = accelerator.unwrap_model(trainer.gpt_model)
                            
                            if args.model_type == 'janus-pro':
                                unwrapped_gpt.save_pretrained(
                                    os.path.join(save_path, "janus_finetuned"), 
                                    is_main_process=True, safe_serialization=True
                                )
                                trainer.processor.save_pretrained(os.path.join(save_path, "janus_finetuned"))
                                if trainer.ema_gpt_model is not None:
                                    trainer.ema_gpt_model.save_pretrained(
                                        os.path.join(save_path, "janus_ema"), 
                                        is_main_process=True, safe_serialization=True
                                    )
                            elif args.model_type == 'llamagen':
                                torch.save(unwrapped_gpt.state_dict(), os.path.join(save_path, "gpt_finetuned.pt"))
                                if trainer.ema_gpt_model is not None:
                                    torch.save(trainer.ema_gpt_model.state_dict(), os.path.join(save_path, "gpt_ema.pt"))

                        if args.train_mode in ['decoder', 'both'] and args.model_type == 'llamagen':
                            unwrapped_vq = accelerator.unwrap_model(trainer.vq_model)
                            torch.save(unwrapped_vq.state_dict(), os.path.join(save_path, "vq_finetuned.pt"))
                            if trainer.ema_vq_model is not None:
                                torch.save(trainer.ema_vq_model.state_dict(), os.path.join(save_path, "vq_ema.pt"))
                                
                        print(f"Saved checkpoint to {save_path}")

                if is_sample_step:
                    torch.cuda.empty_cache()
                    prompts = [
                        "A highly detailed portrait of an elderly man with a deep wrinkles and a white beard, photorealistic, 8k, cinematic lighting.",
                        "A close-up photo of a monarch butterfly on a flower, macro photography, sharp focus, vibrant colors.",
                        "A glass of red wine on a wooden table, sunlight passing through, caustic lighting effects, hyper-realistic.",
                        "A blue apple sitting on a stack of old yellow books.",
                        "A green cube on top of a red cylinder, 3d render, white background.",
                        "A astronaut riding a horse in a photorealistic style.",
                        "Three red balls aligned in a row on a white table.",
                        "A dog sitting to the left of a cat on a park bench.",
                        "A small cottage with a mountain in the background and a river in the foreground.",
                        "A cute panda dressed in a space suit playing an electric guitar on the moon, digital art.",
                        "A cyberpunk city street at night with neon lights and rain reflections, futuristic style.",
                        "An oil painting of a cottage in a forest, strictly in the style of Van Gogh, starry night sky.",
                        "A cat playing with a ball of yarn in a cozy living room, natural lighting, warm tones.",
                        "A close-up photo of a baby panda sleeping on a bed of bamboo, macro photography, sharp focus, vibrant colors.",
                        "A glass of water on a wooden table, sunlight passing through, caustic lighting effects, hyper-realistic.",
                        "A cat playing with a ball of yarn in a cozy living room, natural lighting, warm tones."
                    ]
                    local_prompts = prompts[accelerator.process_index::accelerator.num_processes]
                    if len(local_prompts) > 0:
                        samples = trainer.generate(local_prompts, max_length=args.rollout_length, use_ema=True)
                        if accelerator.is_main_process:
                            os.makedirs(args.output_dir, exist_ok=True)
                        for i, img in enumerate(samples):
                            global_idx = accelerator.process_index + i * accelerator.num_processes
                            img.save(os.path.join(args.output_dir, f"sample_s{global_step}_p{global_idx}.png"))
                    accelerator.wait_for_everyone()

    accelerator.end_training()

if __name__ == "__main__":
    main()
