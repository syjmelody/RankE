import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPModel, CLIPProcessor
import os
import sys
import numpy as np

# --- Optional library imports ---
try:
    import ImageReward
    HAS_IMAGE_REWARD = True
except ImportError:
    HAS_IMAGE_REWARD = False

try:
    from hpsv2.src.open_clip import create_model_and_transforms as hps_create_model_and_transforms
    from hpsv2.src.open_clip import get_tokenizer as hps_get_tokenizer
    from hpsv2.utils import root_path as hps_root_path, hps_version_map
    HAS_HPSV2 = True
except ImportError:
    HAS_HPSV2 = False

# =========================================================
# 1. Distributed synchronization utilities
# =========================================================
class DDPSyncRunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-5, device=None):
        self.mean = torch.zeros(shape, device=device if device else 'cpu')
        self.var = torch.ones(shape, device=device if device else 'cpu')
        self.count = epsilon

    def _gather_data(self, x):
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return x
        original_device = x.device
        if x.device.type != 'cuda': x = x.cuda()
        world_size = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, x)
        return torch.cat(gathered_tensors, dim=0).to(original_device)

    def update(self, x):
        with torch.no_grad():
            x_detached = x.detach()
            x_global = self._gather_data(x_detached)
            
            batch_mean = x_global.mean()
            batch_var = x_global.var(unbiased=False)
            batch_count = x_global.shape[0]
            
            if self.count <= 1e-5:
                self.mean = batch_mean
                self.var = batch_var
                self.count = batch_count
            else:
                delta = batch_mean - self.mean
                tot_count = self.count + batch_count
                new_mean = self.mean + delta * batch_count / tot_count
                m_a = self.var * self.count
                m_b = batch_var * batch_count
                M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
                new_var = M2 / tot_count
                
                self.mean = new_mean
                self.var = new_var
                self.count = tot_count

    def normalize(self, x):
        # Ensure mean/var are on the correct device
        mean = self.mean.to(x.device)
        var = self.var.to(x.device)
        return (x - mean) / torch.sqrt(var + 1e-5)


# =========================================================
# 2. MLP / Aesthetic Predictor
# =========================================================
# --- [Critical fix] Define the MLP class in __main__ to avoid pickle-loading issues ---
# Updated structure matching ava+logos-l14-reluMSE.pt (5-layer layout)
class MLP(nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            # Layer 0
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            # Layer 3
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            # Layer 6 (fix: widen from 16 to 64)
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            # Layer 9 (fix: add the 64 -> 16 layer)
            nn.Linear(64, 16),
            nn.Dropout(0.2),
            nn.ReLU(),
            # Layer 12 (fix: output layer 16 -> 1)
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

class AestheticPredictor(MLP):
    pass


# =========================================================
# 3. MultiRewardManager
# =========================================================
class MultiRewardManager(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.weights = config.reward_weights
        self.normalize_rewards = getattr(config, 'normalize_rewards', True)
        
        self.running_stats = {
            k: DDPSyncRunningMeanStd(device=device) for k, v in self.weights.items() if v != 0
        }

        # --- A. Load CLIP ---
        self.clip_model = None
        if self.weights.get("clip_score", 0.0) != 0 or self.weights.get("aesthetic_score", 0.0) != 0:
            print(">>> [Reward] Loading CLIP Model...")
            clip_path = config.reward_paths.get('clip_score')
            if not clip_path: clip_path = "openai/clip-vit-large-patch14"
            
            try:
                self.clip_model = CLIPModel.from_pretrained(clip_path).to(device).eval()
                self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
                self.clip_model.requires_grad_(False)
                
                # [Fix 1] Explicitly place the model on the target device
                self.register_buffer(
                    'clip_mean', 
                    torch.tensor([0.4814, 0.4578, 0.4082], device=device).view(1,3,1,1)
                )
                self.register_buffer(
                    'clip_std', 
                    torch.tensor([0.2686, 0.2613, 0.2757], device=device).view(1,3,1,1)
                )
            except Exception as e:
                print(f"Error loading CLIP: {e}")

        # --- B. Load Aesthetic ---
        self.aesthetic_model = None
        if self.weights.get("aesthetic_score", 0.0) != 0:
            print(">>> [Reward] Loading Aesthetic Predictor...")
            self.aesthetic_model = MLP(input_size=768).to(device).eval()
            
            aes_path = config.reward_paths.get('aesthetic_score')
            if aes_path and os.path.exists(aes_path):
                try:
                    # Monkey Patch for Pickle
                    if not hasattr(sys.modules['__main__'], 'MLP'):
                        sys.modules['__main__'].MLP = MLP
                    
                    ckpt = torch.load(aes_path, map_location=device)
                    state_dict = ckpt.state_dict() if isinstance(ckpt, nn.Module) else ckpt
                    self.aesthetic_model.load_state_dict(state_dict)
                    print(f">>> [Reward] Aesthetic model loaded successfully")
                except Exception as e:
                    print(f"!!! Error loading Aesthetic Checkpoint: {e}")
                    self.weights["aesthetic_score"] = 0.0
                    self.aesthetic_model = None
            else:
                print(f"Warning: Aesthetic model path not found: {aes_path}")
                self.weights["aesthetic_score"] = 0.0

            if self.aesthetic_model:
                self.aesthetic_model.requires_grad_(False)

        # --- C. Load ImageReward ---
        self.ir_model = None
        if self.weights.get("image_reward", 0.0) != 0 and HAS_IMAGE_REWARD:
            print(">>> [Reward] Loading ImageReward...")
            ir_path = config.reward_paths.get('image_reward')
            try:
                self.ir_model = ImageReward.load(ir_path, device=device)
                self.ir_model.requires_grad_(False)
            except Exception as e:
                print(f"Error: {e}")

        # --- D. Load HPSv2 once for efficient inference ---
        self.hps_model = None
        self.hps_preprocess_val = None
        self.hps_tokenizer = None
        self._hpsv2_error_logged = False
        if self.weights.get("hpsv2", 0.0) != 0:
            if not HAS_HPSV2:
                raise ImportError(
                    "HPSv2 weight is non-zero but hpsv2 module not found. "
                    "Please add evaluations/HPSv2 to PYTHONPATH."
                )
            print(">>> [Reward] Loading HPSv2 Model (ViT-H-14)...")
            try:
                model, _, preprocess_val = hps_create_model_and_transforms(
                    'ViT-H-14', 'laion2B-s32B-b79K',
                    precision='amp', device=device, jit=False,
                    force_quick_gelu=False, force_custom_text=False,
                    force_patch_dropout=False, force_image_size=None,
                    pretrained_image=False, image_mean=None, image_std=None,
                    light_augmentation=True, aug_cfg={},
                    output_dict=True, with_score_predictor=False,
                    with_region_predictor=False
                )
                
                # Load the checkpoint from reward_paths when available, otherwise fall back to Hugging Face
                hps_ckpt_path = config.reward_paths.get('hpsv2')
                if hps_ckpt_path and os.path.isdir(hps_ckpt_path):
                    # If a directory is provided, automatically locate the .pt file inside it
                    for fname in ["HPS_v2.1_compressed.pt", "HPS_v2_compressed.pt"]:
                        candidate = os.path.join(hps_ckpt_path, fname)
                        if os.path.isfile(candidate):
                            hps_ckpt_path = candidate
                            break
                    else:
                        print(f"!!! [Reward] No HPS .pt file found in {hps_ckpt_path}, will download from HuggingFace")
                        hps_ckpt_path = None
                
                if hps_ckpt_path is None or not os.path.isfile(hps_ckpt_path):
                    import huggingface_hub
                    hps_ckpt_path = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.0"])
                
                print(f">>> [Reward] Loading HPSv2 checkpoint: {hps_ckpt_path}")
                checkpoint = torch.load(hps_ckpt_path, map_location=device)
                model.load_state_dict(checkpoint['state_dict'])
                del checkpoint
                
                self.hps_model = model.to(device).eval()
                self.hps_model.requires_grad_(False)
                self.hps_preprocess_val = preprocess_val
                self.hps_tokenizer = hps_get_tokenizer('ViT-H-14')
                print(">>> [Reward] HPSv2 model loaded successfully!")
            except Exception as e:
                print(f"!!! [Reward] Failed to load HPSv2: {e}")
                import traceback; traceback.print_exc()
                raise

    def _preprocess_clip_differentiable(self, images):
        # [Fix 2] Double-check that mean/std tensors are on the same device as images
        # The extra to(images.device) call is intentional as a safety guard
        mean = self.clip_mean.to(images.device)
        std = self.clip_std.to(images.device)
        
        norm_images = (images - mean) / std
        if norm_images.shape[-1] != 224:
            norm_images = F.interpolate(
                norm_images, size=(224, 224), mode='bilinear', align_corners=False
            )
        return norm_images

    def forward(self, images, texts):
        batch_size = images.shape[0]
        device = self.device
        total_reward = torch.zeros(batch_size, device=device)
        rewards_breakdown = {}

        image_features = None
        
        # 1. CLIP & Aesthetic (Differentiable)
        if self.clip_model is not None:
            norm_images = self._preprocess_clip_differentiable(images)
            image_features_raw = self.clip_model.get_image_features(pixel_values=norm_images)
            image_features = image_features_raw / image_features_raw.norm(dim=-1, keepdim=True)

            if self.weights.get("clip_score", 0.0) != 0:
                with torch.no_grad():
                    text_inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
                    text_features = self.clip_model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    if len(texts) != batch_size:
                        repeat_factor = batch_size // len(texts)
                        text_features = text_features.repeat_interleave(repeat_factor, dim=0)

                clip_scores = (image_features * text_features).sum(dim=-1) * 100.0
                rewards_breakdown["clip_score"] = clip_scores

            if self.weights.get("aesthetic_score", 0.0) != 0 and self.aesthetic_model is not None:
                aes_scores = self.aesthetic_model(image_features).squeeze()
                rewards_breakdown["aesthetic_score"] = aes_scores

        # 2. Non-Differentiable Rewards
        with torch.no_grad():
            if self.weights.get("image_reward", 0.0) != 0 and self.ir_model is not None:
                pil_imgs = [to_pil_image(img.float().clamp(0, 1)) for img in images]
                curr_texts = texts if len(texts) == batch_size else [t for t in texts for _ in range(batch_size // len(texts))]
                ir_scores = [self.ir_model.score(prompt, pil_img) for pil_img, prompt in zip(pil_imgs, curr_texts)]
                rewards_breakdown["image_reward"] = torch.tensor(ir_scores, device=device)

            if self.weights.get("hpsv2", 0.0) != 0 and self.hps_model is not None:
                pil_imgs = [to_pil_image(img.float().clamp(0, 1)) for img in images]
                curr_texts = texts if len(texts) == batch_size else [t for t in texts for _ in range(batch_size // len(texts))]
                try:
                    hps_images = torch.stack([self.hps_preprocess_val(img) for img in pil_imgs]).to(device, non_blocking=True)
                    hps_text_tokens = self.hps_tokenizer(curr_texts).to(device, non_blocking=True)
                    
                    with torch.cuda.amp.autocast():
                        outputs = self.hps_model(hps_images, hps_text_tokens)
                        hps_img_feat = outputs["image_features"]
                        hps_txt_feat = outputs["text_features"]
                        logits = hps_img_feat @ hps_txt_feat.T
                        hps_scores = torch.diagonal(logits)
                    
                    rewards_breakdown["hpsv2"] = hps_scores * 100.0
                except Exception as e:
                    if not self._hpsv2_error_logged:
                        print(f"!!! [Reward] HPSv2 scoring error: {e}")
                        import traceback; traceback.print_exc()
                        self._hpsv2_error_logged = True
                    rewards_breakdown["hpsv2"] = torch.zeros(batch_size, device=device)

        # 3. Aggregate
        for name, raw_scores in rewards_breakdown.items():
            weight = self.weights[name]
            if self.normalize_rewards:
                self.running_stats[name].update(raw_scores)
                norm_scores = self.running_stats[name].normalize(raw_scores)
                total_reward = total_reward + weight * norm_scores
            else:
                total_reward = total_reward + weight * raw_scores

        return total_reward, rewards_breakdown
    
    
    
    
# =========================================================
# 4. ProjectedDiscriminatorDINOv3
# =========================================================
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel, ViTModel

class DiscriminatorHead(nn.Module):
    """
    Lightweight convolutional projection head
    Projects backbone feature maps to a 1-channel real/fake prediction map.
    """
    def __init__(self, in_channels, start_kernel_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=start_kernel_size, padding=start_kernel_size//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

class ProjectedDiscriminatorDINOv3(nn.Module):
    """
    Dual-Backbone Projected GAN Discriminator
    Contains two branches:
    1. Texture Branch: EfficientNet-B0 (captures high-frequency texture details)
    2. Semantic Branch: DINOv3 ViT (captures low-frequency semantic consistency)
    """
    def __init__(self, backbone_cnn_name='efficientnet_b0', backbone_dinov3_id='facebook/dinov3-vits16-pretrain-lvd1689m'):
        super().__init__()
        
        print(f">>> [Disc] Initializing DINOv3 Projected GAN")
        
        # ======================================================================
        # 1. Load EfficientNet (Texture Branch)
        # ======================================================================
        # Load a pretrained CNN with timm and use only its feature extractor
        self.backbone_cnn = timm.create_model(backbone_cnn_name, pretrained=True, features_only=True).eval()
        
        # Infer the CNN feature dimension dynamically
        dummy_in = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            cnn_feats = self.backbone_cnn(dummy_in)
            cnn_dims = [f.shape[1] for f in cnn_feats]
            
        # ======================================================================
        # 2. Load DINOv3 (Semantic Branch)
        # ======================================================================
        try:
            self.backbone_vit = AutoModel.from_pretrained(backbone_dinov3_id).eval()
        except (ValueError, KeyError):
            print(f">>> [Disc] AutoModel failed, falling back to ViTModel")
            self.backbone_vit = ViTModel.from_pretrained(backbone_dinov3_id).eval()
            
        self.vit_config = self.backbone_vit.config
        self.vit_embed_dim = self.vit_config.hidden_size
        self.vit_patch_size = self.vit_config.patch_size
        
        # --- [Critical fix] Token-index handling ---
        # Many DINOv3 checkpoints report num_register_tokens=4 in config,
        # but the actual output.last_hidden_state length is often 1025 (1 CLS + 1024 patches) with no register tokens present.
        # That would drop 4 patch tokens when start_index=5, causing a mismatch.
        # Fix: trust the observed patch count first, or force start_index to 1.
        
        self.has_cls = getattr(self.vit_config, "use_cls_token", True)
        self.num_cls = 1 if self.has_cls else 0
        
        # [Fix] Ignore register-token config to avoid mismatches
        # If your model truly emits register tokens, you can restore getattr-based behavior
        self.num_registers = 0 
        
        # Compute the starting token index (usually 1)
        self.patch_start_index = self.num_cls + self.num_registers
        
        print(f">>> [Disc] DINOv3 Config: Patch Size={self.vit_patch_size}, "
              f"Start Index={self.patch_start_index} (Forced Reg=0 to fix mismatch)")

        # Freeze all backbone parameters
        for p in self.backbone_cnn.parameters(): p.requires_grad = False
        for p in self.backbone_vit.parameters(): p.requires_grad = False
        
        # ======================================================================
        # 3. Projection Heads (Multi-Scale)
        # ======================================================================
        self.heads = nn.ModuleList()
        # Head 0: CNN Feature Level 1 (Stride 4)
        self.heads.append(DiscriminatorHead(cnn_dims[1], start_kernel_size=3))
        # Head 1: CNN Feature Level 2 (Stride 8)
        self.heads.append(DiscriminatorHead(cnn_dims[2], start_kernel_size=3))
        # Head 2: ViT Feature (Stride 16)
        self.heads.append(DiscriminatorHead(self.vit_embed_dim, start_kernel_size=1))
        
        self.cnn_select_indices = [1, 2]

        # Image normalization parameters (ImageNet)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def _preprocess(self, x):
        """
        Input: [-1, 1] -> Output: Normalized for ImageNet
        """
        x = (x + 1.0) * 0.5
        x = (x - self.mean) / self.std
        return x

    def forward(self, x):
        # x: [B, 3, H, W], range [-1, 1]
        x_norm = self._preprocess(x)
        features = []
        B, C, H, W = x_norm.shape

        # --- 1. EfficientNet Forward ---
        with torch.no_grad():
            cnn_out = self.backbone_cnn(x_norm)
            for idx in self.cnn_select_indices:
                features.append(cnn_out[idx])
        
        # --- 2. DINOv3 Forward (With Robust Slicing) ---
        with torch.no_grad():
            # Handle non-divisible resolution padding
            pad_h = (self.vit_patch_size - H % self.vit_patch_size) % self.vit_patch_size
            pad_w = (self.vit_patch_size - W % self.vit_patch_size) % self.vit_patch_size
            
            if pad_h > 0 or pad_w > 0:
                x_vit = F.pad(x_norm, (0, pad_w, 0, pad_h), mode='reflect')
            else:
                x_vit = x_norm
            
            # Forward Pass
            outputs = self.backbone_vit(x_vit, interpolate_pos_encoding=True)
            feat = outputs.last_hidden_state # [B, SeqLen, D]
            
            # Calculate Expected Grid Size
            h_grid = x_vit.shape[2] // self.vit_patch_size
            w_grid = x_vit.shape[3] // self.vit_patch_size
            expected_patches = h_grid * w_grid
            
            # --- Robust Slicing Strategy ---
            # Strategy A: slice using the configured start_index
            patch_tokens = feat[:, self.patch_start_index:, :]
            
            # Strategy B: if lengths mismatch, enable automatic fallback correction
            # This resolves cases like "Expected 1024, got 1020"
            if patch_tokens.shape[1] != expected_patches:
                # Print the warning only once to avoid log spam
                # print(f"DINO Mismatch: Seq={feat.shape[1]}, Exp={expected_patches}. Using tail cropping.")
                patch_tokens = feat[:, -expected_patches:, :]

            # Reshape to Feature Map: [B, H*W, D] -> [B, D, H, W]
            dino_feat = patch_tokens.transpose(1, 2).reshape(B, self.vit_embed_dim, h_grid, w_grid)
            
            # Crop back padding if needed
            if pad_h > 0 or pad_w > 0:
                h_valid = H // self.vit_patch_size
                w_valid = W // self.vit_patch_size
                dino_feat = dino_feat[:, :, :h_valid, :w_valid]

            features.append(dino_feat)

        # --- 3. Multi-Scale Heads Forward ---
        logits_list = []
        for i, head in enumerate(self.heads):
            # i=0: CNN stride 4
            # i=1: CNN stride 8
            # i=2: ViT stride 16
            logits = head(features[i])
            logits_list.append(logits)
            
        return logits_list
    



# =========================================================
# 5. HybridVQLoss (supports multiple discriminator types)
# =========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer.tokenizer_image.lpips import LPIPS

try:
    from tokenizer.tokenizer_image.discriminator_patchgan import NLayerDiscriminator
except ImportError:
    NLayerDiscriminator = None

def adopt_weight(weight, global_step, threshold=0, value=0.):
    """Time scheduler: return value (usually 0) before threshold, then return weight"""
    if global_step < threshold:
        return value
    return weight

def hinge_d_loss(logits_real, logits_fake, weights=None):
    """Hinge Loss for Discriminator"""
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = F.relu(1. + logits_fake)
    
    if weights is not None:
        # RankE: weight fake samples so higher-weight cases push the discriminator harder
        # Assumes weights have shape [B] and need broadcasting to match logits
        # Logits may have shape [B, 1, H, W]
        w_broad = weights.view(-1, 1, 1, 1) if logits_fake.dim() == 4 else weights.view(-1, 1)
        # If spatial logits do not match weight shape (for example with PatchGAN), reduce them before weighting
        if logits_fake.shape[0] == weights.shape[0]:
            loss_fake = (loss_fake * w_broad).mean()
        else:
            loss_fake = loss_fake.mean()
    else:
        loss_fake = torch.mean(loss_fake)
        
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def hinge_gen_loss(logits_fake, weights=None):
    """Hinge Loss for Generator"""
    if weights is not None:
        # RankE: weight generated samples so higher-reward cases push the generator harder
        w_broad = weights.view(-1, 1, 1, 1) if logits_fake.dim() == 4 else weights.view(-1, 1)
        if logits_fake.shape[0] == weights.shape[0]:
            return -(logits_fake * w_broad).mean()
    return -torch.mean(logits_fake)

class HybridVQLoss(nn.Module):
    def __init__(self, disc_start, disc_weight=1.0, perceptual_weight=1.0, reconstruction_weight=1.0, 
                 disc_type="projected", dino_path=None, use_adaptive_weight=True):
        super().__init__()
        self.rec_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.disc_weight = disc_weight
        self.disc_start = disc_start
        self.disc_type = disc_type
        self.use_adaptive_weight = use_adaptive_weight

        print(f">>> [HybridVQLoss] Config: Type={disc_type}, Adaptive={use_adaptive_weight}, Start={disc_start}")

        # =========================================================
        # 1. Initialize the discriminator
        # =========================================================
        if self.disc_type == "projected":
            self.discriminator = ProjectedDiscriminatorDINOv3(backbone_dinov3_id=dino_path)
        
        elif self.disc_type == "patchgan":
            if NLayerDiscriminator is None:
                raise ImportError("NLayerDiscriminator module not found.")
            print(f">>> [HybridVQLoss] Initializing Original VQGAN Discriminator (PatchGAN)")
            self.discriminator = NLayerDiscriminator(input_nc=3, n_layers=3, use_actnorm=False, ndf=64)
        
        else:
            raise ValueError(f"Unknown discriminator type: {self.disc_type}")

        self.discriminator.train()
        
        # =========================================================
        # 2. Initialize perceptual loss (LPIPS)
        # =========================================================
        self.perceptual_loss = LPIPS().eval()
        for p in self.perceptual_loss.parameters():
            p.requires_grad = False

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        """
        Compute adaptive adversarial weight lambda = grad(rec) / grad(adv)
        """
        fallback = torch.tensor(self.disc_weight, device=nll_loss.device)
        if last_layer is None:
            return fallback

        # Ensure all related tensors require gradients
        if not nll_loss.requires_grad or not g_loss.requires_grad:
            return fallback
        if not last_layer.requires_grad:
            return fallback

        try:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads_tuple = torch.autograd.grad(g_loss, last_layer, retain_graph=True, allow_unused=True)
            g_grads = g_grads_tuple[0]

            if g_grads is None:
                return fallback

            d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
            d_weight = d_weight * self.disc_weight
            return d_weight
        except RuntimeError:
            return fallback
        

    def forward(self, inputs, reconstructions, optimizer_idx, global_step, 
                weights=None, last_layer=None):
        """
        Args:
            inputs: real images [B, 3, H, W]
            reconstructions: reconstructed images [B, 3, H, W]
            optimizer_idx: 0 (Generator), 1 (Discriminator)
            global_step: current training step
            weights: optional RankE weights [B]
            last_layer: optional decoder final-layer convolution weights used for adaptive weighting
        """
        
        # Helper: ensure outputs are always handled as a list (PatchGAN vs. ProjectedGAN)
        def ensure_list(x):
            return x if isinstance(x, (list, tuple)) else [x]

        # =========================================================
        # 1. Generator Update (Decoder)
        # =========================================================
        if optimizer_idx == 0:
            # A. Reconstruction Loss (L1)
            rec_loss = torch.abs(inputs - reconstructions)
            
            # B. Perceptual Loss (LPIPS)
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs, reconstructions)
                rec_loss = rec_loss + self.perceptual_weight * p_loss
            else:
                p_loss = torch.tensor(0.0)

            # RankE Weighting for NLL (Negative Log Likelihood)
            if weights is not None:
                # Ensure weights can broadcast to [B, C, H, W]
                w_broad = weights.view(-1, 1, 1, 1)
                nll_loss = (rec_loss * w_broad).mean()
            else:
                nll_loss = rec_loss.mean()

            # C. GAN Loss (Generator side)
            # Start adversarial computation only after disc_start
            disc_factor = adopt_weight(self.disc_weight, global_step, threshold=self.disc_start)
            
            g_loss = torch.tensor(0.0, device=inputs.device)
            adaptive_weight_val = 1.0

            if disc_factor > 0:
                logits_fake = self.discriminator(reconstructions)
                logits_fake_list = ensure_list(logits_fake)
                
                g_loss_val = 0.0
                for logits in logits_fake_list:
                    g_loss_val += hinge_gen_loss(logits, weights)
                
                # Average the multi-scale losses
                g_loss = g_loss_val / len(logits_fake_list)

                # D. Adaptive weight calculation (key step)
                if self.use_adaptive_weight and self.training and g_loss.requires_grad:
                    # Use the original unweighted loss terms when computing gradients
                    adaptive_weight_val = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
                else:
                    adaptive_weight_val = disc_factor

            # E. Total Loss
            total_loss = nll_loss + adaptive_weight_val * g_loss
            
            return total_loss, {
                "val/rec_loss": rec_loss.mean().item(),
                "val/p_loss": p_loss.mean().item(),
                "val/nll_loss": nll_loss.item(),
                "val/g_loss": g_loss.item(),
                "val/d_weight": adaptive_weight_val
            }

        # =========================================================
        # 2. Discriminator Update
        # =========================================================
        if optimizer_idx == 1:
            disc_factor = adopt_weight(self.disc_weight, global_step, threshold=self.disc_start)
            
            # Return zero gradient before adversarial training starts
            if disc_factor == 0:
                return torch.tensor(0.0, device=inputs.device, requires_grad=True), {}

            # Detach inputs so we don't backprop to Generator
            logits_real = self.discriminator(inputs.detach())
            logits_fake = self.discriminator(reconstructions.detach())
            
            logits_real_list = ensure_list(logits_real)
            logits_fake_list = ensure_list(logits_fake)
            
            d_loss = 0.0
            for l_real, l_fake in zip(logits_real_list, logits_fake_list):
                d_loss += hinge_d_loss(l_real, l_fake, weights)
            
            d_loss = (d_loss / len(logits_real_list)) * disc_factor
            
            return d_loss, {"val/d_loss": d_loss.item(), "val/logits_real": logits_real_list[0].mean().item(), "val/logits_fake": logits_fake_list[0].mean().item()}