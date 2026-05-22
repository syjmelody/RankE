# Script Reference

This document gives a short English reference for each public script under `scripts/`.

---

## LlamaGen

### Training

#### `scripts/training_llamagen/train_clip_both.sh`
- Purpose: LlamaGen post-training with the BLIP3O / CLIP-style reward setup
- Main inputs: `SFT_SOURCE_RUN`, `SFT_SOURCE_STEP`, `SCALING_SIZE`
- Main outputs: `${PROJECT_OUTPUT_ROOT}/ranke_llamagen/<run_name>/checkpoint_*`

#### `scripts/training_llamagen/train_hps_both.sh`
- Purpose: LlamaGen post-training with the HPDv2 / HPSv2-style reward setup
- Main inputs: `SFT_SOURCE_RUN`, `SFT_SOURCE_STEP`, `SCALING_SIZE`
- Main outputs: `${PROJECT_OUTPUT_ROOT}/ranke_llamagen/<run_name>/checkpoint_*`

### Sampling

#### `scripts/sample_llamagen/sample_clip_coco.sh`
- Purpose: sample COCO images from a LlamaGen checkpoint
- Inputs: `RUN_NAME`, `STEPS`, `COMBO_ID`, `CFG_LIST`
- Outputs: `${PROJECT_OUTPUT_ROOT}/ranke_llamagen/<run_name>/samples_coco/...`

#### `scripts/sample_llamagen/sample_clip_geneval.sh`
- Purpose: sample GenEval images from a LlamaGen checkpoint
- Inputs: `RUN_NAME`, `STEPS`, `COMBO_ID`, `CFG_SCALE`
- Outputs: `${PROJECT_OUTPUT_ROOT}/ranke_llamagen/<run_name>/samples_geneval/...`

#### `scripts/sample_llamagen/sample_hpsv2.sh`
- Purpose: sample HPSv2 images from a LlamaGen checkpoint
- Inputs: `RUN_NAME`, `STEPS`, `COMBO_ID`, `CFG_LIST`
- Outputs: `${PROJECT_OUTPUT_ROOT}/ranke_llamagen/<run_name>/samples_hpsv2/...`

#### `scripts/sample_llamagen/sample_hps_geneval.sh`
- Purpose: run GenEval sampling for a LlamaGen model trained on the HPDv2 path
- Inputs: `RUN_NAME`, `STEPS`, `COMBO_ID`, `CFG_SCALE`
- Outputs: `${PROJECT_OUTPUT_ROOT}/ranke_llamagen/<run_name>/samples_geneval/...`

### Evaluation

#### `scripts/eval_llamagen/eval_coco.sh`
- Purpose: evaluate LlamaGen COCO samples
- Inputs: `RUN_NAME`, `STEPS`, `COMBO_ID`, `CFG_SCALE`
- Outputs: `score.txt` inside sample directories plus a run-level summary text file

#### `scripts/eval_llamagen/eval_geneval.sh`
- Purpose: evaluate LlamaGen GenEval samples from the BLIP3O-trained path
- Inputs: `RUN_NAME`, `STEPS`, `COMBO_ID`, `CFG_SCALE`
- Outputs: `results.jsonl` / `final_score.txt` inside sample directories plus a run-level summary text file

#### `scripts/eval_llamagen/eval_hpsv2.sh`
- Purpose: evaluate LlamaGen HPSv2 samples
- Inputs: `RUN_NAME`, `STEPS`, `COMBO_ID`, `CFG_LIST`
- Outputs: `score.txt` inside sample directories plus a run-level summary text file

#### `scripts/eval_llamagen/eval_hps_geneval.sh`
- Purpose: evaluate GenEval results for a LlamaGen model trained on the HPDv2 path
- Inputs: `RUN_NAME`, `STEPS`, `COMBO_ID`, `CFG_SCALE`
- Outputs: `results.jsonl` / `final_score.txt` inside sample directories plus a run-level summary text file

---

## Janus

### Training

#### `scripts/training_janus/train_clip_both.sh`
- Purpose: Janus post-training with the BLIP3O / CLIP-style reward setup
- Main inputs: `MODEL_TYPE`, `SFT_SOURCE_RUN`, `SFT_SOURCE_STEP`, `SCALING_SIZE`
- Main outputs: `${PROJECT_OUTPUT_ROOT}/ranke_janus/<run_name>/checkpoint_*`

#### `scripts/training_janus/train_hps_both.sh`
- Purpose: Janus post-training with the HPDv2 / HPSv2-style reward setup
- Main inputs: `MODEL_TYPE`, `SFT_SOURCE_RUN`, `SFT_SOURCE_STEP`, `SCALING_SIZE`
- Main outputs: `${PROJECT_OUTPUT_ROOT}/ranke_janus/<run_name>/checkpoint_*`

### Sampling

#### `scripts/sample_janus/sample_coco.sh`
- Purpose: sample COCO images from a Janus checkpoint
- Inputs: `RUN_NAME`, `STEPS`, `COMBO_ID`, `CFG_LIST`
- Outputs: `${PROJECT_OUTPUT_ROOT}/ranke_janus/<run_name>/samples_coco/...`

#### `scripts/sample_janus/sample_clip_geneval.sh`
- Purpose: sample GenEval images from a Janus checkpoint trained on the BLIP3O path
- Inputs: `RUN_NAME`, `STEPS`, `COMBO_ID`, `CFG_SCALE`
- Outputs: `${PROJECT_OUTPUT_ROOT}/ranke_janus/<run_name>/samples_geneval/...`

#### `scripts/sample_janus/sample_hpsv2.sh`
- Purpose: sample HPSv2 images from a Janus checkpoint
- Inputs: `RUN_NAME`, `STEPS`, `COMBO_ID`, `CFG_LIST`
- Outputs: `${PROJECT_OUTPUT_ROOT}/ranke_janus/<run_name>/samples_hpsv2/...`

#### `scripts/sample_janus/sample_hps_geneval.sh`
- Purpose: run GenEval sampling for a Janus model trained on the HPDv2 path
- Inputs: `RUN_NAME`, `STEPS`, `COMBO_ID`, `CFG_SCALE`
- Outputs: `${PROJECT_OUTPUT_ROOT}/ranke_janus/<run_name>/samples_geneval/...`

### Evaluation

#### `scripts/eval_janus/eval_coco.sh`
- Purpose: evaluate Janus COCO samples
- Inputs: `RUN_NAME`, `STEPS`, `COMBO_ID`, `CFG_LIST`
- Outputs: `score.txt` inside sample directories plus a run-level summary text file

#### `scripts/eval_janus/eval_clip_geneval.sh`
- Purpose: evaluate Janus GenEval samples from the BLIP3O-trained path
- Inputs: `RUN_NAME`, `STEPS`, `COMBO_ID`, `CFG_SCALE`
- Outputs: `results.jsonl` / `final_score.txt` inside sample directories plus a run-level summary text file

#### `scripts/eval_janus/eval_hpsv2.sh`
- Purpose: evaluate Janus HPSv2 samples
- Inputs: `RUN_NAME`, `STEPS`, `COMBO_ID`, `CFG_LIST`
- Outputs: `score.txt` inside sample directories plus a run-level summary text file

#### `scripts/eval_janus/eval_hps_geneval.sh`
- Purpose: evaluate GenEval results for a Janus model trained on the HPDv2 path
- Inputs: `RUN_NAME`, `STEPS`, `COMBO_ID`, `CFG_SCALE`
- Outputs: `results.jsonl` / `final_score.txt` inside sample directories plus a run-level summary text file

---

## Suggested execution order

### LlamaGen
1. training script
2. matching sampling script
3. matching evaluation script

### Janus
1. training script
2. matching sampling script
3. matching evaluation script
