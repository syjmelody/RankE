# RankE

This repository contains the cleaned open-source release of the core **RankE post-training pipelines** built on top of:

- **LlamaGen**
- **Janus-Pro**

The release focuses on the code required to:

- start from an existing **base** or **SFT** checkpoint,
- run **post-training**,
- generate samples for **COCO / GenEval / HPSv2**,
- run the corresponding **evaluation** pipelines.

> This repository intentionally keeps the core training, sampling, and evaluation code needed for the paper workflow.
> Large datasets, checkpoints, logs, analysis notebooks, and older experimental branches are excluded.

---

## Repository layout

```text
RankE/
├── autoregressive/              # Core LlamaGen-side model, training, and sampling code
├── tokenizer/                   # Tokenizer, VQ, and loss modules used by LlamaGen-side code
├── language/                    # Text encoding utilities (T5)
├── dataset/                     # Training data loading utilities
├── janus/                       # Core Janus-side model, training, and sampling code
├── evaluations/                 # COCO / GenEval / HPSv2 evaluation code
├── scripts/
│   ├── training_llamagen/       # LlamaGen training scripts
│   ├── sample_llamagen/         # LlamaGen sampling scripts
│   ├── eval_llamagen/           # LlamaGen evaluation scripts
│   ├── training_janus/          # Janus training scripts
│   ├── sample_janus/            # Janus sampling scripts
│   └── eval_janus/              # Janus evaluation scripts
├── configs/
│   └── config.env.example       # Local environment/path template
├── docs/
│   ├── USAGE.md                 # End-to-end usage guide
│   └── SCRIPT_REFERENCE.md      # Short reference for each public script
├── requirements.txt             # Main dependencies
├── requirements_janus.txt       # Additional Janus-related dependency reference
└── OPEN_SOURCE_SCOPE.md         # Scope of this release
```

---

## Included core capabilities

### LlamaGen pipeline
- RankE post-training
- BLIP3O-based and HPDv2-based training entry points
- COCO / GenEval / HPSv2 sampling
- COCO / GenEval / HPSv2 evaluation

### Janus pipeline
- RankE post-training
- BLIP3O-based and HPDv2-based training entry points
- COCO / GenEval / HPSv2 sampling
- COCO / GenEval / HPSv2 evaluation

---

## Quick start

### 1. Install dependencies

Use a dedicated Python or Conda environment.

```bash
pip install -r requirements.txt
```

If you mainly run the Janus pipeline, also review or install:

```bash
pip install -r requirements_janus.txt
```

> The GenEval, HPSv2, and Janus stacks may require version-specific adjustments depending on your CUDA, PyTorch, and `transformers` setup.

### 2. Configure local paths

Copy the template:

```bash
cp configs/config.env.example configs/config.env
```

Then edit the main variables in `configs/config.env`, such as:

- `PROJECT_OUTPUT_ROOT`
- `PRETRAINED_ROOT`
- `VQ_CKPT_PATH`
- `GPT_CKPT_PATH_STAGE1`
- `JANUS_MODEL_PATH`
- `SCALING_BLIP3O_ROOT`
- `HPDV2_TRAIN_ROOT`
- `HPDV2_EVAL_ROOT`
- `COCO_REF_DIR`
- `GENEVAL_MASK2FORMER_PATH`
- `HPSV2_MODEL_PATH`

All cleaned scripts read `configs/config.env` by default.
You can also override it temporarily:

```bash
export RANKE_CONFIG_ENV=/your/path/config.env
```

---

## Public script entry points

### LlamaGen

**Training**
- `scripts/training_llamagen/train_clip_both.sh`
- `scripts/training_llamagen/train_hps_both.sh`

**Sampling**
- `scripts/sample_llamagen/sample_clip_coco.sh`
- `scripts/sample_llamagen/sample_clip_geneval.sh`
- `scripts/sample_llamagen/sample_hpsv2.sh`
- `scripts/sample_llamagen/sample_hps_geneval.sh`

**Evaluation**
- `scripts/eval_llamagen/eval_coco.sh`
- `scripts/eval_llamagen/eval_geneval.sh`
- `scripts/eval_llamagen/eval_hpsv2.sh`
- `scripts/eval_llamagen/eval_hps_geneval.sh`

### Janus

**Training**
- `scripts/training_janus/train_clip_both.sh`
- `scripts/training_janus/train_hps_both.sh`

**Sampling**
- `scripts/sample_janus/sample_coco.sh`
- `scripts/sample_janus/sample_clip_geneval.sh`
- `scripts/sample_janus/sample_hpsv2.sh`
- `scripts/sample_janus/sample_hps_geneval.sh`

**Evaluation**
- `scripts/eval_janus/eval_coco.sh`
- `scripts/eval_janus/eval_clip_geneval.sh`
- `scripts/eval_janus/eval_hpsv2.sh`
- `scripts/eval_janus/eval_hps_geneval.sh`

---

## Recommended workflow

### LlamaGen
1. Edit the training script.
2. Set `SFT_SOURCE_RUN`, `SFT_SOURCE_STEP`, `SCALING_SIZE`, reward weights, and optimizer settings.
3. Launch training.
4. Copy the produced `RUN_NAME` into the sampling script.
5. Run sampling.
6. Run evaluation.

### Janus
1. Edit the training script.
2. Set `MODEL_TYPE="janus-pro"`, `SFT_SOURCE_RUN`, `SFT_SOURCE_STEP`, `SCALING_SIZE`, and the training hyperparameters.
3. Launch training.
4. Copy the produced `RUN_NAME` into the sampling script.
5. Run sampling.
6. Run evaluation.

See also:

- [`docs/USAGE.md`](docs/USAGE.md)
- [`docs/SCRIPT_REFERENCE.md`](docs/SCRIPT_REFERENCE.md)

---

## Notes

### Script style
The current scripts still follow an experiment-oriented shell-script style. In practice, you will usually edit variables such as:

- `RUN_NAME`
- `STEPS`
- `CFG_SCALE` / `CFG_LIST`
- `COMBO_ID`

before running them.

### Scope of the codebase
This release focuses on:

- starting from an existing base or SFT checkpoint,
- running RankE post-training,
- running downstream sampling and evaluation.

It does **not** aim to provide a full pretraining or full SFT reproduction stack.

### Output directories
By default, scripts write outputs under:

```bash
${PROJECT_OUTPUT_ROOT}/...
```

including checkpoints, samples, and evaluation summaries.

---

## Licenses

- LlamaGen-related code: see `LICENSE`
- Janus-related code: see `LICENSE-CODE` and `LICENSE-MODEL`

Please review the licenses of upstream code, checkpoints, and models before use.

---

## Open-source scope

See:

- [`OPEN_SOURCE_SCOPE.md`](OPEN_SOURCE_SCOPE.md)
