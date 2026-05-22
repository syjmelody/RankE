# RankE Usage Guide

This document explains how to run the cleaned public RankE repository with the two supported pipelines:

- **RankE + LlamaGen**
- **RankE + Janus-Pro**

---

## 1. Three-stage workflow

Both pipelines follow the same high-level structure:

1. **Training / post-training**
   - Input: a base model or an SFT checkpoint
   - Output: a new `RUN_NAME` directory containing one or more `checkpoint_xxx` subdirectories

2. **Sampling**
   - Input: a selected checkpoint from a training run
   - Output: generated image directories such as:
     - `samples_coco/...`
     - `samples_geneval/...`
     - `samples_hpsv2/...`

3. **Evaluation**
   - Input: a generated image directory
   - Output:
     - per-sample `score.txt` or `final_score.txt`
     - run-level summary text files

---

## 2. Configuration

Copy the template first:

```bash
cp configs/config.env.example configs/config.env
```

### Important variables to check

#### General
- `CODE_ROOT`
- `STORAGE_ROOT`
- `PROJECT_OUTPUT_ROOT`
- `PRETRAINED_ROOT`

#### LlamaGen
- `VQ_CKPT_PATH`
- `GPT_CKPT_PATH_STAGE1`
- `GPT_CKPT_PATH_STAGE2`
- `T5_PATH`

#### Janus
- `JANUS_MODEL_PATH`

#### Reward and evaluation models
- `CLIP_PATH`
- `DINO_PATH`
- `AES_REW_PATH`
- `HPSV2_MODEL_PATH`
- `GENEVAL_MASK2FORMER_PATH`

#### Data
- `SCALING_BLIP3O_ROOT`
- `HPDV2_TRAIN_ROOT`
- `HPDV2_EVAL_ROOT`
- `COCO_REF_DIR`
- `GENEVAL_PROMPTS_FILE`

---

## 3. LlamaGen pipeline

### Training scripts

**BLIP3O / CLIP reward path**
```bash
bash scripts/training_llamagen/train_clip_both.sh
```

**HPDv2 / HPSv2 reward path**
```bash
bash scripts/training_llamagen/train_hps_both.sh
```

### Common variables to edit before training

At the top of the script, the most common edits are:

- `SFT_SOURCE_RUN`
- `SFT_SOURCE_STEP`
- `RESUME_DIR`
- `TRAIN_MODE`
- `SCALING_SIZE`
- `LR_GPT`
- `LR_DECODER`
- `LR_DISC`
- `R_CLIP / R_HPSV2 / R_AESTHETIC / R_IMAGE_REWARD`
- `SAVE_INTERVAL`
- `SAMPLING_INTERVAL`

### Typical output structure

Training outputs are usually written under:

```bash
${PROJECT_OUTPUT_ROOT}/ranke_llamagen/<RUN_NAME>/
```

Typical contents include:

- `checkpoint_500/`
- `checkpoint_1000/`
- ...
- `gpt_finetuned.pt`
- `gpt_ema.pt`
- `vq_finetuned.pt`
- `vq_ema.pt`

### Sampling scripts

**COCO**
```bash
bash scripts/sample_llamagen/sample_clip_coco.sh
```

**GenEval (BLIP3O-trained run)**
```bash
bash scripts/sample_llamagen/sample_clip_geneval.sh
```

**HPSv2**
```bash
bash scripts/sample_llamagen/sample_hpsv2.sh
```

**GenEval (HPDv2-trained run)**
```bash
bash scripts/sample_llamagen/sample_hps_geneval.sh
```

### Common variables to edit before sampling

- `RUN_NAME`
- `STEPS`
- `COMBO_ID`
- `CFG_SCALE` or `CFG_LIST`

### LlamaGen `COMBO_ID`

Common combinations used by the LlamaGen scripts:

- `0`: base model
- `1`: GPT online + VQ online
- `2`: GPT online + VQ EMA
- `3`: GPT EMA + VQ online
- `4`: GPT EMA + VQ EMA
- `5`: GPT online + VQ base

### Evaluation scripts

**COCO**
```bash
bash scripts/eval_llamagen/eval_coco.sh
```

**GenEval (BLIP3O-trained run)**
```bash
bash scripts/eval_llamagen/eval_geneval.sh
```

**HPSv2**
```bash
bash scripts/eval_llamagen/eval_hpsv2.sh
```

**GenEval (HPDv2-trained run)**
```bash
bash scripts/eval_llamagen/eval_hps_geneval.sh
```

Before evaluation, make sure:

- the sampling directory for the target `RUN_NAME` already exists,
- `COCO_REF_DIR` is correct,
- `GENEVAL_MASK2FORMER_PATH` is correct,
- `HPSV2_MODEL_PATH` is correct.

---

## 4. Janus pipeline

### Training scripts

**BLIP3O / CLIP reward path**
```bash
bash scripts/training_janus/train_clip_both.sh
```

**HPDv2 / HPSv2 reward path**
```bash
bash scripts/training_janus/train_hps_both.sh
```

### Common variables to edit before training

- `MODEL_TYPE` (normally `janus-pro`)
- `SFT_SOURCE_RUN`
- `SFT_SOURCE_STEP`
- `RESUME_DIR`
- `TRAIN_MODE`
- `SCALING_SIZE`
- `LR_GPT`
- `LR_DECODER`
- `LR_DISC`
- `R_CLIP / R_HPSV2`
- `GROUP_SIZE`
- `SAVE_INTERVAL`

### Typical output structure

Training outputs are usually written under:

```bash
${PROJECT_OUTPUT_ROOT}/ranke_janus/<RUN_NAME>/
```

Typical contents include:

- `checkpoint_xxx/`
- `janus_finetuned/`
- `janus_ema/`

Unlike LlamaGen, Janus stores a full model directory rather than separate GPT and VQ `.pt` files.

### Sampling scripts

**COCO**
```bash
bash scripts/sample_janus/sample_coco.sh
```

**GenEval (BLIP3O-trained run)**
```bash
bash scripts/sample_janus/sample_clip_geneval.sh
```

**HPSv2**
```bash
bash scripts/sample_janus/sample_hpsv2.sh
```

**GenEval (HPDv2-trained run)**
```bash
bash scripts/sample_janus/sample_hps_geneval.sh
```

### Common variables to edit before sampling

- `RUN_NAME`
- `STEPS`
- `COMBO_ID`
- `CFG_SCALE` or `CFG_LIST`

### Janus `COMBO_ID`

The Janus scripts use a simpler variant layout:

- `0`: base model
- `1`: online model
- `2`: EMA model

### Evaluation scripts

**COCO**
```bash
bash scripts/eval_janus/eval_coco.sh
```

**GenEval (BLIP3O-trained run)**
```bash
bash scripts/eval_janus/eval_clip_geneval.sh
```

**HPSv2**
```bash
bash scripts/eval_janus/eval_hpsv2.sh
```

**GenEval (HPDv2-trained run)**
```bash
bash scripts/eval_janus/eval_hps_geneval.sh
```

---

## 5. Naming conventions in the scripts

- `clip`: mainly the BLIP3O / CLIP reward direction
- `hps`: mainly the HPDv2 / HPSv2 reward direction
- `both`: GPT and decoder are both updated
- `coco`: COCO-based evaluation path
- `geneval`: compositional evaluation on GenEval
- `hpsv2`: HPSv2 preference/alignment evaluation

---

## 6. Variables you will edit most often

### Training scripts
- `SFT_SOURCE_RUN`
- `SFT_SOURCE_STEP`
- `RESUME_DIR`
- `SCALING_SIZE`
- reward weights
- learning rates
- save/sampling intervals

### Sampling scripts
- `RUN_NAME`
- `STEPS`
- `COMBO_ID`
- `CFG_SCALE` or `CFG_LIST`

### Evaluation scripts
- `RUN_NAME`
- `STEPS`
- `COMBO_ID`
- `CFG_SCALE` or `CFG_LIST`
- dataset/reference paths

---

## 7. Minimal example workflows

### Example A: LlamaGen + BLIP3O
1. Edit `scripts/training_llamagen/train_clip_both.sh`
2. Launch training
3. Record the resulting `RUN_NAME`
4. Edit `scripts/sample_llamagen/sample_clip_geneval.sh` or `sample_clip_coco.sh`
5. Run sampling
6. Edit the matching evaluation script
7. Run evaluation

### Example B: Janus + HPDv2
1. Edit `scripts/training_janus/train_hps_both.sh`
2. Launch training
3. Record the resulting `RUN_NAME`
4. Edit `scripts/sample_janus/sample_hpsv2.sh` or `sample_hps_geneval.sh`
5. Run sampling
6. Edit the matching evaluation script
7. Run evaluation

---

## 8. FAQ

### Why must sampling run before evaluation?
Because the COCO, GenEval, and HPSv2 evaluation scripts score generated image directories, not raw checkpoints.

### Why do some scripts still contain default experimental values?
These scripts were cleaned from real research workflows. The defaults are kept only as runnable examples. You should replace them with your own `RUN_NAME`, steps, and local paths.

### Why do Janus and LlamaGen checkpoints look different?
LlamaGen stores GPT and VQ weights as separate files. Janus stores a model directory, so the Janus sampling scripts use a `base_model_path + target_ckpt_path` loading pattern.

### Should I run `bash script.sh` or `sbatch script.sh`?
Most scripts are still written in a cluster-oriented style and are best suited to `sbatch`. Some also work for local debugging with `bash`, but you may need to adjust launcher or environment settings for your own setup.

---

## 9. Suggested next cleanup steps

If you want to keep refining the public release, good next steps are:

1. add template variants for each script family,
2. provide an `environment.yml`,
3. add dataset preparation notes,
4. add checkpoint download/setup notes.
