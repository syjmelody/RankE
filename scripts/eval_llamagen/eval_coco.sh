#!/bin/bash
#SBATCH --job-name=ranke_llamagen_eval_coco
#SBATCH --output=logs/eval_coco_%j.out
#SBATCH --error=logs/eval_coco_%j.err
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
# [1] Core configuration (USER CONFIG)
# ==============================================================================

# --- A. Run source (must match the sampling script) ---
RUN_NAME=""   # Fill in a run name that has already been sampled; leave empty to use base-model logic

# Checkpoint steps to sample
STEPS=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000)

# --- B. Combination mode selection ---
# 1: GPT(Online) + VQ(Online)
# 2: GPT(Online) + VQ(EMA)
# 3: GPT(EMA)    + VQ(Online)
# 4: GPT(EMA)    + VQ(EMA)
# 5: GPT(Online) + VQ(Base)   <--- policy-only test mode
# 0: Pure base mode (active automatically when RUN_NAME="")
COMBO_ID=1

# --- C. Inference configuration ---
CFG_SCALE=6.0
IMAGE_SIZE=256
BATCH_SIZE=128     # Evaluation batch size (larger is faster, subject to memory limits)

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

# Resolve tags
case "$COMBO_ID" in
    0) GPT_TAG="gpt-base"; VQ_EXPECTED="vq-base" ;;
    1) GPT_TAG="gpt-onl";  VQ_EXPECTED="vq-onl"  ;;
    2) GPT_TAG="gpt-onl";  VQ_EXPECTED="vq-ema"  ;;
    3) GPT_TAG="gpt-ema";  VQ_EXPECTED="vq-onl"  ;;
    4) GPT_TAG="gpt-ema";  VQ_EXPECTED="vq-ema"  ;;
    5) GPT_TAG="gpt-onl";  VQ_EXPECTED="vq-base" ;; # Added for the public release
    *) echo "Error: Invalid COMBO_ID $COMBO_ID"; exit 1 ;;
esac

# Unified root-directory logic (aligned with the sampling script)
TASK_ROOT="${PROJECT_OUTPUT_ROOT}/ranke_llamagen"

if [ -z "$RUN_NAME" ]; then
    EXP_ROOT="${TASK_ROOT}/base_model_evaluation"
else
    EXP_ROOT="${TASK_ROOT}/${RUN_NAME}"
fi

SAMPLES_ROOT="${EXP_ROOT}/samples_coco"
EVAL_SCRIPT="${CODE_ROOT}/evaluations/t2i/evaluation.py"
REF_DIR="${COCO_REF_DIR:-${STORAGE_ROOT}/dataset/coco2014}" 

# Global summary file path (includes CFG to keep results separated)
CONSOLIDATED_SCORE_FILE="${EXP_ROOT}/results_coco_cfg${CFG_SCALE}.txt"

export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH}"
mkdir -p "$EXP_ROOT"

echo "========================================================================"
echo ">>> COCO Evaluation Pipeline | Combo Key: $COMBO_ID"
echo ">>> Looking for: $GPT_TAG + $VQ_EXPECTED (or vq-base fallback)"
echo ">>> CFG Scale  : $CFG_SCALE"
echo ">>> Steps      : ${STEPS[*]}"
echo "========================================================================"

# ==============================================================================
# [3] Evaluation loop
# ==============================================================================

# Handle pure base-mode logic
if [ -z "$RUN_NAME" ]; then
    loop_steps=("base")
else
    loop_steps=("${STEPS[@]}")
fi

for step in "${loop_steps[@]}"; do
    
    echo ">>> ---------------------------------------------------"
    echo ">>> Processing Step: $step"

    # [Core logic] Smart two-stage lookup
    # -----------------------------------------------------
    if [ "$step" == "base" ]; then
        TARGET_DIR="${SAMPLES_ROOT}/baseline_gpt-base_vq-base_cfg${CFG_SCALE}"
        FOUND_TYPE="Base Model"
    else
        # Mode A: search for the exact matching directory (for example ..._gpt-onl_vq-ema_...)
        DIR_PATTERN_PRIMARY="sample_step${step}_${GPT_TAG}_${VQ_EXPECTED}_cfg${CFG_SCALE}"
        TARGET_DIR=$(find "$SAMPLES_ROOT" -maxdepth 1 -type d -name "$DIR_PATTERN_PRIMARY" | head -n 1)
        FOUND_TYPE="Primary Match"

        # Mode B: if no exact match is found and the expected target is not base, search for a fallback directory
        if [ -z "$TARGET_DIR" ] && [ "$VQ_EXPECTED" != "vq-base" ]; then
            DIR_PATTERN_FALLBACK="sample_step${step}_${GPT_TAG}_vq-base_cfg${CFG_SCALE}"
            TARGET_DIR=$(find "$SAMPLES_ROOT" -maxdepth 1 -type d -name "$DIR_PATTERN_FALLBACK" | head -n 1)
            FOUND_TYPE="Fallback Match (vq-base)"
        fi
    fi

    # Validate the target directory
    if [ -z "$TARGET_DIR" ] || [ ! -d "$TARGET_DIR" ]; then
        echo ">>> [Skip] Directory not found for Step $step (CFG $CFG_SCALE)."
        continue
    fi

    SCORE_FILE="${TARGET_DIR}/score.txt"
    EVALUATED_JUST_NOW=false

    # If it has not been evaluated yet, run the official evaluation script
    if [ ! -f "$SCORE_FILE" ]; then
        echo ">>> Found Target ($FOUND_TYPE): $(basename "$TARGET_DIR")"
        echo ">>> Running Evaluation..."

        python "$EVAL_SCRIPT" \
            --fake_dir "$TARGET_DIR" \
            --ref_dir "$REF_DIR" \
            --ref_data "coco2014" \
            --ref_type "val2014" \
            --how_many 30000 \
            --eval_res $IMAGE_SIZE \
            --batch_size $BATCH_SIZE \
            --clip_model4eval "ViT-B/32"

        EVALUATED_JUST_NOW=true
        echo ">>> Step $step Evaluation Finished."
    else
        echo ">>> [Check] Already evaluated ($FOUND_TYPE)."
        echo ">>> Path: $(basename "$TARGET_DIR")"
        EVALUATED_JUST_NOW=true # Set to true so the result is appended to the summary file
    fi
    
    # -----------------------------------------------------
    # Show results and append them to the summary file
    # -----------------------------------------------------
    if [ -f "$SCORE_FILE" ] && [ "$EVALUATED_JUST_NOW" = true ]; then
        echo ">>> Result for Step $step:"
        cat "$SCORE_FILE"
        
        # Append results with timestamp, step, and config details for easy comparison
        {
            echo "=========================================================="
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Step: $step | Combo: $GPT_TAG + $VQ_EXPECTED | CFG: $CFG_SCALE"
            echo "Dir: $(basename "$TARGET_DIR")"
            echo "----------------------------------------------------------"
            cat "$SCORE_FILE"
            echo ""
        } >> "$CONSOLIDATED_SCORE_FILE"
        
        echo ">>> [Success] Result appended to $CONSOLIDATED_SCORE_FILE"
    elif [ ! -f "$SCORE_FILE" ]; then
        echo ">>> [Warning] Evaluation completed but $SCORE_FILE was not found!"
    fi

    # In base mode the weights are fixed, so evaluate once and stop
    if [ "$step" == "base" ] && [ -f "$SCORE_FILE" ]; then
        echo ">>> [Notice] Base mode evaluation complete."
        break
    fi

done

echo ">>> ---------------------------------------------------"
echo ">>> All Done. Consolidated results saved to:"
echo ">>> $CONSOLIDATED_SCORE_FILE"