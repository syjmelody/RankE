import os
import json
import glob
import math
import torch
import webdataset as wds
import io
import sys
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image, ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================================================================
# 1. Dataset registry
# ==============================================================================
JSON_ROOT = os.getenv("RANKE_JSON_ROOT", "")
WDS_ROOT = os.getenv("RANKE_WDS_ROOT", "")

DATASET_CATALOG = {
    "t2i_2m": {
        "type": "json",
        "path": f"{JSON_ROOT}/sft_data.jsonl",
        "image_root": ""
    },
    "blip3o": {
        "type": "webdataset",
        "root": f"{WDS_ROOT}/BLIP3o-60k",
        "total_samples": 60000,
        "default_mix": "all:1.0",
        "shards": {
            "dalle3": "dalle3.tar",
            "geneval": "geneval_train.tar",
            "human": "human_gestures.tar",
            "journey": "journeyDB.tar",
            "coco": "mscoco_human.tar",
            "obj1": "object_1.tar",
            "obj2": "object_2.tar",
            "occu1": "occupation_1.tar",
            "occu2": "occupation_2.tar",
            "text1": "text_1.tar",
            "text2": "text_2.tar"
        }
    },
    "scaling_simple": {
        "type": "webdataset_simple", 
        "root": "", 
        "total_samples": 0 
    },
}

def get_dataset_config(name):
    if name not in DATASET_CATALOG:
        sys.stderr.write(f"Warning: Dataset '{name}' not found in catalog. Using default simple config.\n")
        return {"type": "webdataset_simple", "root": ""}
    return DATASET_CATALOG[name]


# ==============================================================================
# 2. Global helper functions (must be picklable)
# ==============================================================================

# [Critical fix] Replace lambda src: src to keep multiprocessing serialization safe
def identity_splitter(src):
    return src

def sanitize_keys_global(sample):
    """Normalize sample keys"""
    new_sample = {}
    for key, value in sample.items():
        key_lower = key.lower()
        if key_lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            new_sample["jpg"] = value
        elif key_lower.endswith((".txt", ".caption", ".json")):
            new_sample["txt"] = value
        else:
            new_sample[key] = value
    return new_sample

class ScalingPreprocessor:
    """Robust preprocessing helper"""
    def __init__(self, transform, image_size=256):
        self.transform = transform
        self.image_size = image_size

    def __call__(self, sample):
        # 1. Unpack
        try:
            raw_img, raw_txt = sample
        except Exception:
            return torch.zeros((3, self.image_size, self.image_size)), ""

        # 2. Image processing
        pixel_values = None
        try:
            if isinstance(raw_img, bytes):
                pil_img = Image.open(io.BytesIO(raw_img)).convert("RGB")
            elif isinstance(raw_img, Image.Image):
                pil_img = raw_img.convert("RGB")
            else:
                pil_img = Image.open(str(raw_img)).convert("RGB")
            
            pixel_values = self.transform(pil_img)
        except Exception:
            pixel_values = torch.zeros((3, self.image_size, self.image_size))

        # 3. Text processing
        caption = ""
        try:
            if isinstance(raw_txt, bytes):
                caption = raw_txt.decode("utf-8", errors="ignore")
            else:
                caption = str(raw_txt)
            caption = caption.strip()
        except:
            caption = ""
            
        if not isinstance(pixel_values, torch.Tensor):
             pixel_values = torch.zeros((3, self.image_size, self.image_size))

        return pixel_values, caption


# ==============================================================================
# 3. Dataset factory and collate functions
# ==============================================================================

class SFTDataset: 
    @staticmethod
    def collate_fn(batch):
        """
        [Defensive fallback] Automatically repair malformed data and avoid raising exceptions
        """
        cleaned_batch_imgs = []
        cleaned_batch_txts = []
        fallback_size = 256 
        
        for i, item in enumerate(batch):
            # Check structure: must be a tuple/list with length >= 2
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                sys.stderr.write(f"[SFTDataset Warn] Skipped bad item type: {type(item)}\n")
                continue 
                
            img, txt = item[0], item[1]
            
            # --- Image validation ---
            if isinstance(img, torch.Tensor):
                fallback_size = img.shape[-1]
                cleaned_batch_imgs.append(img)
            else:
                # Replace non-tensor images (for example strings) with black images
                sys.stderr.write(f"[SFTDataset FIXED] Replaced bad image (type: {type(img)}) with black tensor.\n")
                cleaned_batch_imgs.append(torch.zeros((3, fallback_size, fallback_size)))

            # --- Text validation ---
            cleaned_batch_txts.append(str(txt) if txt is not None else "")

        # Fallback: if the whole batch becomes empty
        if not cleaned_batch_imgs:
            sys.stderr.write("[SFTDataset Alert] Empty batch detected. Returning dummy batch.\n")
            return torch.zeros((1, 3, 256, 256)), ["error"]

        # Stack tensors
        try:
            pixel_values = torch.stack(cleaned_batch_imgs)
        except Exception as e:
            sys.stderr.write(f"[SFTDataset Stack Fail] {e}. Using full dummy batch.\n")
            pixel_values = torch.zeros((len(cleaned_batch_imgs), 3, fallback_size, fallback_size))
            
        return pixel_values, cleaned_batch_txts


class _JSONDataset(Dataset):
    def __init__(self, path, transform=None, image_root=""):
        self.data = []
        files = [p.strip() for p in path.split(',')]
        print(f">>> [_JSONDataset] Scanning files...")
        for p in files:
            if not os.path.exists(p):
                print(f"Warning: File not found {p}")
                continue
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            self.data.append(json.loads(line))
                        except:
                            pass
        self.transform = transform
        self.image_root = image_root
        print(f">>> [_JSONDataset] Total loaded: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['image_path']
        if self.image_root and not img_path.startswith('/'):
            img_path = os.path.join(self.image_root, img_path)
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        text = item.get('text', '')
        return image, text

class _WebDatasetBuilder:
    def __init__(self, config, mix_override, transform):
        self.transform = transform
        base_path = config['root']
        shard_map = config.get('shards', {})
        total_samples = config.get('total_samples', 10000)
        mix_config = mix_override if mix_override else config['default_mix']
        self.sources = [] 
        
        if mix_config.lower().startswith("all"):
            if shard_map:
                for fname in shard_map.values():
                    fpath = os.path.join(base_path, fname)
                    if os.path.exists(fpath): self.sources.append(([fpath], 1.0))
            else:
                all_tars = sorted(glob.glob(os.path.join(base_path, "*.tar")))
                if all_tars: self.sources.append((all_tars, 1.0))
        else:
            for part in mix_config.split(','):
                key, weight = part.split(':')
                weight = float(weight)
                key = key.strip().lower()
                fname = shard_map.get(key, key)
                search_path = os.path.join(base_path, fname)
                matched_files = sorted(glob.glob(search_path))
                if matched_files: self.sources.append((matched_files, weight))
        
        total_w = sum(w for _, w in self.sources) if self.sources else 1.0
        self.weights = [w / total_w for _, w in self.sources]
        self.total_samples_est = total_samples

    def _preprocess(self, sample):
        image, caption = sample
        try:
            if self.transform:
                pixel_values = self.transform(image) if self.transform else image
        except:
            pixel_values = torch.zeros((3, 512, 512))
        
        if isinstance(caption, bytes): caption = caption.decode("utf-8")
        return pixel_values, str(caption).strip()

    def get_loader(self, local_batch_size, num_workers, world_size):
        datasets = []
        for urls, _ in self.sources:
            ds = (
                wds.WebDataset(urls, resampled=True, nodesplitter=wds.split_by_node)
                .shuffle(1000)
                .decode("pil")
                .to_tuple("jpg", "txt")
                .map(self._preprocess)
            )
            datasets.append(ds)
        mixed_dataset = wds.RandomMix(datasets, self.weights) if len(datasets) > 1 else datasets[0]
        loader = DataLoader(mixed_dataset, batch_size=local_batch_size, num_workers=num_workers, 
                          pin_memory=True, drop_last=True, collate_fn=SFTDataset.collate_fn)
        return loader



def build_dataloader(args, transform, world_size, rank):
    dataset_name = getattr(args, 'dataset_name', 't2i_2m')
    config = get_dataset_config(dataset_name).copy() 
    override_path = getattr(args, 'data_path', None)
    if override_path:
        config['root' if 'webdataset' in config['type'] else 'path'] = override_path
    
    dataset_type = config['type']
    print(f">>> [DataFactory] Building dataset: '{dataset_name}' (Type: {dataset_type})")

    micro_bs = int(args.global_batch_size // world_size // args.gradient_accumulation_steps)
    if micro_bs < 1: micro_bs = 1

    # =======================================================
    # [Patched branch] Scaling Simple mode
    # =======================================================
    if dataset_type == 'webdataset_simple':
        tar_path = config['root']
        n_samples = 20000 
        for k in ["5k", "10k", "15k", "20k"]:
            if k in tar_path: n_samples = int(k.replace("k", "000"))
        
        print(f">>> [ScalingLoader] Target: {tar_path} (Est: {n_samples})")
        image_size = getattr(args, 'image_size', 256) 
        preprocessor = ScalingPreprocessor(transform=transform, image_size=image_size)

        # 1. Build the dataset (without batched loader wrappers)
        dataset = wds.WebDataset(
            tar_path, 
            resampled=True, 
            nodesplitter=identity_splitter, 
            shardshuffle=True, 
            handler=wds.warn_and_continue
        ) \
            .shuffle(1000) \
            .map(sanitize_keys_global) \
            .to_tuple("jpg", "txt") \
            .map(preprocessor) 
        
        # 2. [Core change] Use a standard DataLoader instead of WebLoader.batched
        # This way collate_fn receives [(img, txt), (img, txt), ...] instead of transposed data
        loader = DataLoader(
            dataset, 
            batch_size=micro_bs, 
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=SFTDataset.collate_fn 
        )
        
        steps_per_epoch = max(1, n_samples // args.global_batch_size)
        return loader, None, steps_per_epoch

    # =======================================================
    # [Other branches remain unchanged]
    # =======================================================
    elif dataset_type == 'json':
        dataset = _JSONDataset(path=config['path'], transform=transform, image_root=config.get('image_root', ''))
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.global_seed)
        loader = DataLoader(dataset, batch_size=micro_bs, shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=SFTDataset.collate_fn)
        steps_per_epoch = len(loader) // args.gradient_accumulation_steps
        return loader, sampler, steps_per_epoch

    elif dataset_type == 'webdataset':
        mix_override = getattr(args, 'wds_mix_config', None)
        builder = _WebDatasetBuilder(config=config, mix_override=mix_override, transform=transform)
        loader = builder.get_loader(local_batch_size=micro_bs, num_workers=args.num_workers, world_size=world_size)
        total_samples = config['total_samples']
        steps_per_epoch = total_samples // args.global_batch_size // args.gradient_accumulation_steps
        return loader, None, steps_per_epoch

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")