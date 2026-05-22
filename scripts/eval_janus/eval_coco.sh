#!/bin/bash
#SBATCH --job-name=ranke_janus_eval_coco
#SBATCH --output=logs/coco_eval_janus_%j.out
#SBATCH --error=logs/coco_eval_janus_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1             # FID/CLIP evaluation usually needs only 1 GPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00

set -e
export PYTHONUNBUFFERED=1

# ==============================================================================
# [1] Core configuration (must match the sampling script)
# ==============================================================================

# --- A. Run source ---
RUN_NAME=""   # Fill in a run name that has already been sampled; leave empty to use base-model logic

# Checkpoint steps to evaluate (space-separated)
STEPS=(500 1000 1500 2000 2500 3000 3500 4000)

# --- B. Combination mode selection (aligned with the sampling script) ---
# 0: Base model
# 1: janus_finetuned (Online)
# 2: janus_ema       (EMA)
COMBO_ID=1

# --- C. Inference configuration ---
CFG_LIST=(5.0)
IMAGE_SIZE=384       
BATCH_SIZE=64        
NUM_SAMPLES=30000

# ==============================================================================
# [2] Resolve combination tag and environment paths
# ==============================================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
CONFIG_FILE="${RANKE_CONFIG_ENV:-${REPO_ROOT}/configs/config.env}"
CONFIG_EXAMPLE="${REPO_ROOT}/configs/config.env.example"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
elif [ -f "$CONFIG_EXAMPLE" ]; then
    echo ">>> [Notice] Using config.env.example. Please create configs/config.env for local paths."
    source "$CONFIG_EXAMPLE"
else
    echo ">>> [Error] config.env not found under ${REPO_ROOT}/configs" && exit 1
fi

case "$COMBO_ID" in
    0) COMBO_TAG="janus-base" ;;
    1) COMBO_TAG="janus-onl"  ;;
    2) COMBO_TAG="janus-ema"  ;;
    *) echo "Error: Invalid COMBO_ID $COMBO_ID"; exit 1 ;;
esac

TASK_ROOT="${PROJECT_OUTPUT_ROOT}/ranke_janus"

if [ -z "$RUN_NAME" ]; then
    EXP_ROOT="${TASK_ROOT}/base_model_evaluation"
else
    EXP_ROOT="${TASK_ROOT}/${RUN_NAME}"
fi

SAMPLES_ROOT="${EXP_ROOT}/samples_coco"

# [Core] Use the official evaluation script
EVAL_SCRIPT="${CODE_ROOT}/evaluations/t2i/evaluation.py"
REF_DIR="${COCO_GT_DIR}" 

export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH}"

echo "========================================================================"
echo ">>> COCO Evaluation Pipeline (Janus) | Combo: $COMBO_TAG"
echo ">>> CFG List   : ${CFG_LIST[*]}"
echo ">>> Steps      : ${STEPS[*]}"
echo ">>> Target Root: $SAMPLES_ROOT"
echo "========================================================================"

# ==============================================================================
# [3] Evaluation loop
# ==============================================================================

if [ -z "$RUN_NAME" ]; then
    loop_steps=("base")
else
    loop_steps=("${STEPS[@]}")
fi

for CURRENT_CFG in "${CFG_LIST[@]}"; do
    CONSOLIDATED_SCORE_FILE="${EXP_ROOT}/results_coco_cfg${CURRENT_CFG}.txt"

    for step in "${loop_steps[@]}"; do

        echo ">>> ---------------------------------------------------"
        echo ">>> Processing Step: $step | CFG: $CURRENT_CFG"

        if [ "$step" == "base" ]; then
            TARGET_DIR="${SAMPLES_ROOT}/baseline_janus-base_cfg${CURRENT_CFG}"
            FOUND_TYPE="Base Model"
        else
            DIR_PATTERN="sample_step${step}_${COMBO_TAG}_cfg${CURRENT_CFG}"
            TARGET_DIR="${SAMPLES_ROOT}/${DIR_PATTERN}"
            FOUND_TYPE="Primary Match"
        fi

        if [ ! -d "$TARGET_DIR" ]; then
            echo ">>> [Skip] Directory not found: $(basename "$TARGET_DIR")"
            continue
        fi

        # ==========================================================================
        # [Critical patch] Generate captions.txt on the fly to avoid evaluation.py failures
        # ==========================================================================
        if [ ! -f "$TARGET_DIR/captions.txt" ]; then
            echo ">>> [Patch] captions.txt not found in $TARGET_DIR. Auto-generating..."
            python -c "
import pandas as pd
import os

prompt_file = '${EVAL_PROMPT_FILE_COCO}'
num_samples = ${NUM_SAMPLES}
target_dir = '${TARGET_DIR}'

try:
    df = pd.read_csv(prompt_file)
    cols = [c.lower() for c in df.columns]
    col_name = 'caption' if 'caption' in cols else ('prompt' if 'prompt' in cols else df.columns[0])
    prompts = df[df.columns[cols.index(col_name)]].astype(str).tolist()
except Exception as e:
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

if len(prompts) < num_samples:
    prompts = (prompts * (num_samples // len(prompts) + 1))[:num_samples]
else:
    prompts = prompts[:num_samples]

with open(os.path.join(target_dir, 'captions.txt'), 'w', encoding='utf-8') as f:
    for p in prompts:
        f.write(p.strip() + '\n')
print(f'>>> [Patch] Successfully generated captions.txt with {len(prompts)} lines.')
"
        fi
        # ==========================================================================

        SCORE_FILE="${TARGET_DIR}/score.txt"
        EVALUATED_JUST_NOW=false

        if [ ! -f "$SCORE_FILE" ]; then
            echo ">>> Found Target ($FOUND_TYPE): $(basename "$TARGET_DIR")"
            echo ">>> Running Evaluation..."

            python "$EVAL_SCRIPT" \
                --fake_dir "$TARGET_DIR" \
                --ref_dir "$REF_DIR" \
                --ref_data "coco2014" \
                --ref_type "val2014" \
                --how_many $NUM_SAMPLES \
                --eval_res $IMAGE_SIZE \
                --batch_size $BATCH_SIZE \
                --clip_model4eval "ViT-B/32"

            EVALUATED_JUST_NOW=true
            echo ">>> Step $step Evaluation Finished."
        else
            echo ">>> [Check] Already evaluated ($FOUND_TYPE)."
            echo ">>> Path: $(basename "$TARGET_DIR")"
            EVALUATED_JUST_NOW=true
        fi

        if [ -f "$SCORE_FILE" ] && [ "$EVALUATED_JUST_NOW" = true ]; then
            echo ">>> Result for Step $step:"
            cat "$SCORE_FILE"

            {
                echo "=========================================================="
                echo "[$(date +'%Y-%m-%d %H:%M:%S')] Step: $step | Combo: $COMBO_TAG | CFG: $CURRENT_CFG"
                echo "Dir: $(basename "$TARGET_DIR")"
                echo "----------------------------------------------------------"
                cat "$SCORE_FILE"
                echo ""
            } >> "$CONSOLIDATED_SCORE_FILE"

            echo ">>> [Success] Result appended to $CONSOLIDATED_SCORE_FILE"
        elif [ ! -f "$SCORE_FILE" ]; then
            echo ">>> [Warning] Evaluation completed but $SCORE_FILE was not found!"
        fi

        if [ "$step" == "base" ] && [ -f "$SCORE_FILE" ]; then
            echo ">>> [Notice] Base mode evaluation complete."
            break
        fi

    done
done

echo ">>> ---------------------------------------------------"
echo ">>> All Done. Consolidated results saved to:"
echo ">>> ${EXP_ROOT}/results_coco_cfg*.txt"
