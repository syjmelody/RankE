# RankE — End-to-End Post-Training for Discrete Text-to-Image Generation with Decoder Co-Evolution

> **TL;DR** — RankE is an end-to-end post-training framework for discrete autoregressive text-to-image generation that **co-evolves the decoder together with the generator**, instead of keeping the decoder frozen throughout post-training.

<p align="center">
  <img src="assets/teaser.png" alt="RankE teaser" width="100%">
</p>

RankE revisits post-training for **discrete autoregressive text-to-image models**.
Instead of optimizing only the generator against text-image rewards, RankE jointly updates the **generator**, **decoder**, and **discriminator-based supervision path**, enabling better end-to-end adaptation after SFT/base pretraining.

**[Paper](https://arxiv.org/abs/2605.21195), [Usage Guide](docs/USAGE.md), [Script Reference](docs/SCRIPT_REFERENCE.md)**

Siyong Jian, Siyuan Li, Luyuan Zhang, Zedong Wang, Xin Jin, Ying Li, Cheng Tan, Huan Wang <br>

## News

- 🚀 [May 2026] RankE paper released on arXiv: [2605.21195](https://arxiv.org/abs/2605.21195).
- 🧩 This repository provides public code paths for **LlamaGen** and **Janus-Pro** post-training, sampling, and evaluation.

## Installation

> [!TIP]
> **Quick Start** — if your environment already has a working PyTorch/CUDA stack, you can usually start with:
>
> ```bash
> pip install -r requirements.txt
> ```
>
> For the **Janus-Pro** pipeline, also install:
>
> ```bash
> pip install -r requirements_janus.txt
> ```

Full setup from a fresh environment:

```bash
# create / activate your own Python or Conda environment first
pip install -r requirements.txt
pip install -r requirements_janus.txt   # optional, only if you plan to run Janus-Pro
```

## Checkpoints, datasets, and local paths

RankE is a **code-only release**. This repository does **not** ship model checkpoints, benchmark data, or reward-model weights. You should prepare the required upstream assets locally and point the scripts to them through `configs/config.env`.

Copy the template first:

```bash
cp configs/config.env.example configs/config.env
```

Important variables are grouped as follows:

- **General**
  - `CODE_ROOT`
  - `STORAGE_ROOT`
  - `PROJECT_OUTPUT_ROOT`
  - `PRETRAINED_ROOT`
- **LlamaGen**
  - `VQ_CKPT_PATH`
  - `GPT_CKPT_PATH_STAGE1`
  - `GPT_CKPT_PATH_STAGE2`
  - `T5_PATH`
- **Janus-Pro**
  - `JANUS_MODEL_PATH`
- **Reward / evaluation models**
  - `CLIP_PATH`
  - `DINO_PATH`
  - `AES_REW_PATH`
  - `HPSV2_MODEL_PATH`
  - `GENEVAL_MASK2FORMER_PATH`
- **Datasets / prompts**
  - `SCALING_BLIP3O_ROOT`
  - `HPDV2_TRAIN_ROOT`
  - `HPDV2_EVAL_ROOT`
  - `COCO_REF_DIR`
  - `GENEVAL_PROMPTS_FILE`

All public scripts read `configs/config.env` by default. To use a custom file:

```bash
export RANKE_CONFIG_ENV=/path/to/config.env
```

## Running RankE

RankE currently provides two post-training backbones and a shared **three-stage workflow**:

1. **Post-training**
2. **Sampling**
3. **Evaluation**

### Supported pipelines

| Backbone | Post-training scripts | Sampling targets | Evaluation targets |
|----------|------------------------|------------------|--------------------|
| LlamaGen | `scripts/training_llamagen/*.sh` | COCO / GenEval / HPSv2 | COCO / GenEval / HPSv2 |
| Janus-Pro | `scripts/training_janus/*.sh` | COCO / GenEval / HPSv2 | COCO / GenEval / HPSv2 |

### 1) Post-training

#### LlamaGen

```bash
# BLIP3O / CLIP reward path
bash scripts/training_llamagen/train_clip_both.sh

# HPDv2 / HPSv2 reward path
bash scripts/training_llamagen/train_hps_both.sh
```

Most commonly edited variables before launch:

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

#### Janus-Pro

```bash
# BLIP3O / CLIP reward path
bash scripts/training_janus/train_clip_both.sh

# HPDv2 / HPSv2 reward path
bash scripts/training_janus/train_hps_both.sh
```

Most commonly edited variables before launch:

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

### 2) Sampling

#### LlamaGen

```bash
# COCO
bash scripts/sample_llamagen/sample_clip_coco.sh

# GenEval (BLIP3O-trained run)
bash scripts/sample_llamagen/sample_clip_geneval.sh

# HPSv2
bash scripts/sample_llamagen/sample_hpsv2.sh

# GenEval (HPDv2-trained run)
bash scripts/sample_llamagen/sample_hps_geneval.sh
```

Common variables to edit:

- `RUN_NAME`
- `STEPS`
- `COMBO_ID`
- `CFG_SCALE` or `CFG_LIST`

Common `COMBO_ID` values used by the LlamaGen scripts:

- `0`: base model
- `1`: GPT online + VQ online
- `2`: GPT online + VQ EMA
- `3`: GPT EMA + VQ online
- `4`: GPT EMA + VQ EMA
- `5`: GPT online + VQ base

#### Janus-Pro

```bash
# COCO
bash scripts/sample_janus/sample_coco.sh

# GenEval (BLIP3O-trained run)
bash scripts/sample_janus/sample_clip_geneval.sh

# HPSv2
bash scripts/sample_janus/sample_hpsv2.sh

# GenEval (HPDv2-trained run)
bash scripts/sample_janus/sample_hps_geneval.sh
```

Common variables to edit:

- `RUN_NAME`
- `STEPS`
- `COMBO_ID`
- `CFG_SCALE` or `CFG_LIST`

### 3) Evaluation

#### LlamaGen

```bash
bash scripts/eval_llamagen/eval_coco.sh
bash scripts/eval_llamagen/eval_geneval.sh
bash scripts/eval_llamagen/eval_hpsv2.sh
bash scripts/eval_llamagen/eval_hps_geneval.sh
```

#### Janus-Pro

```bash
bash scripts/eval_janus/eval_coco.sh
bash scripts/eval_janus/eval_clip_geneval.sh
bash scripts/eval_janus/eval_hpsv2.sh
bash scripts/eval_janus/eval_hps_geneval.sh
```

Before evaluation, make sure:

- the sampling directory for the target `RUN_NAME` already exists,
- `COCO_REF_DIR` is correct,
- `GENEVAL_MASK2FORMER_PATH` is correct,
- `HPSV2_MODEL_PATH` is correct.

## Output layout

Typical outputs are written under `${PROJECT_OUTPUT_ROOT}`.

- **LlamaGen**
  - `${PROJECT_OUTPUT_ROOT}/ranke_llamagen/<RUN_NAME>/checkpoint_*`
  - `${PROJECT_OUTPUT_ROOT}/ranke_llamagen/<RUN_NAME>/samples_coco/...`
  - `${PROJECT_OUTPUT_ROOT}/ranke_llamagen/<RUN_NAME>/samples_geneval/...`
  - `${PROJECT_OUTPUT_ROOT}/ranke_llamagen/<RUN_NAME>/samples_hpsv2/...`
- **Janus-Pro**
  - `${PROJECT_OUTPUT_ROOT}/ranke_janus/<RUN_NAME>/checkpoint_*`
  - `${PROJECT_OUTPUT_ROOT}/ranke_janus/<RUN_NAME>/samples_coco/...`
  - `${PROJECT_OUTPUT_ROOT}/ranke_janus/<RUN_NAME>/samples_geneval/...`
  - `${PROJECT_OUTPUT_ROOT}/ranke_janus/<RUN_NAME>/samples_hpsv2/...`

For a more complete walkthrough, see:

- [`docs/USAGE.md`](docs/USAGE.md)
- [`docs/SCRIPT_REFERENCE.md`](docs/SCRIPT_REFERENCE.md)

## Repository layout

```text
RankE/
├── autoregressive/              # LlamaGen-side model, training, and sampling code
├── tokenizer/                   # Tokenizer, VQ, and loss modules
├── language/                    # Text encoding utilities
├── dataset/                     # Training data loading utilities
├── janus/                       # Janus-side model, training, and sampling code
├── evaluations/                 # COCO / GenEval / HPSv2 evaluation code
├── scripts/
│   ├── training_llamagen/
│   ├── sample_llamagen/
│   ├── eval_llamagen/
│   ├── training_janus/
│   ├── sample_janus/
│   └── eval_janus/
├── configs/
│   └── config.env.example       # Local environment/path template
├── docs/
│   ├── USAGE.md                 # End-to-end usage guide
│   └── SCRIPT_REFERENCE.md      # Public script reference
├── assets/
│   └── teaser.png               # README teaser figure
├── requirements.txt
├── requirements_janus.txt
└── README.md
```

## License

This repository is distributed under the [MIT License](LICENSE).

Please also check the original licenses and usage terms of any upstream repositories, checkpoints, datasets, and evaluation assets you use together with RankE.

## Acknowledgments

This release includes RankE pipelines built on top of:

- [**LlamaGen**](https://github.com/FoundationVision/LlamaGen)
- [**Janus-Pro**](https://github.com/deepseek-ai/janus)

## Citation

```bibtex
@article{jian2026ranke,
  title={RankE: End-to-End Post-Training for Discrete Text-to-Image Generation with Decoder Co-Evolution},
  author={Jian, Siyong and Li, Siyuan and Zhang, Luyuan and Wang, Zedong and Jin, Xin and Li, Ying and Tan, Cheng and Wang, Huan},
  journal={arXiv preprint arXiv:2605.21195},
  year={2026}
}
```
