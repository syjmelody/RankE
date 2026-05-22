#!/bin/bash
#SBATCH --job-name=ranke_llamagen_eval_hpsv2
#SBATCH --output=logs/hpsv2_eval_%j.out
#SBATCH --error=logs/hpsv2_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1           # Evaluation usually needs only 1 GPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00

set -e
export PYTHONUNBUFFERED=1

# ==============================================================================
# [1] Core configuration (USER CONFIG)
# ==============================================================================
RUN_NAME=""   # Fill in a run name that has already been sampled; leave empty to use base-model logic
STEPS=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000)
# Support multiple CFG values, aligned with the sampling scripts
CFG_LIST=(6.0)
COMBO_ID=2

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

# [Local resource paths (required for offline evaluation)]
HPDV2_PATH="${HPDV2_EVAL_ROOT}"
HPS_MODEL_PATH="${HPSV2_MODEL_PATH}"

# Add your local HPSv2 repository to PYTHONPATH
LOCAL_HPSV2_REPO="${CODE_ROOT}/evaluations/HPSv2"
export PYTHONPATH="${LOCAL_HPSV2_REPO}:${CODE_ROOT}:${PYTHONPATH}"

EVAL_SCRIPT="${CODE_ROOT}/evaluations/HPSv2/eval_hpsv2.py"
TASK_ROOT="${PROJECT_OUTPUT_ROOT}/ranke_llamagen"

if [ -z "$RUN_NAME" ]; then
    EXP_ROOT="${TASK_ROOT}/base_model_evaluation"
else
    EXP_ROOT="${TASK_ROOT}/${RUN_NAME}"
fi

SAMPLES_ROOT="${EXP_ROOT}/samples_hpsv2"

# Resolve tags
case "$COMBO_ID" in
    0) GPT_TAG="gpt-base"; VQ_EXPECTED="vq-base" ;;
    1) GPT_TAG="gpt-onl";  VQ_EXPECTED="vq-onl"  ;;
    2) GPT_TAG="gpt-onl";  VQ_EXPECTED="vq-ema"  ;;
    3) GPT_TAG="gpt-ema";  VQ_EXPECTED="vq-onl"  ;;
    4) GPT_TAG="gpt-ema";  VQ_EXPECTED="vq-ema"  ;;
    5) GPT_TAG="gpt-onl";  VQ_EXPECTED="vq-base" ;;
esac

echo "========================================================================"
echo ">>> Offline HPSv2 Evaluation Pipeline | Combo Key: $COMBO_ID"
echo ">>> Looking for: $GPT_TAG + $VQ_EXPECTED (or vq-base fallback)"
echo ">>> CFG List   : ${CFG_LIST[*]}"
echo ">>> Steps      : ${STEPS[*]}"
echo "========================================================================"

# ==============================================================================
# [3] Evaluation loop
# ==============================================================================
loop_steps=("${STEPS[@]}")
if [ "$COMBO_ID" -eq 0 ]; then
    loop_steps=("base")
elif [ -z "$RUN_NAME" ]; then
    loop_steps=("base")
fi

# Outer loop over CFG values
for CURRENT_CFG in "${CFG_LIST[@]}"; do
    
    CONSOLIDATED_SCORE_FILE="${EXP_ROOT}/results_hpsv2_cfg${CURRENT_CFG}.txt"

    for step in "${loop_steps[@]}"; do
        echo ">>> ---------------------------------------------------"
        echo ">>> Processing Step: $step | CFG: $CURRENT_CFG"

        # Smart two-stage lookup
        if [ "$step" == "base" ]; then
            TARGET_DIR="${SAMPLES_ROOT}/baseline_gpt-base_vq-base_cfg${CURRENT_CFG}"
            FOUND_TYPE="Base Model"
        else
            DIR_PATTERN_PRIMARY="sample_step${step}_${GPT_TAG}_${VQ_EXPECTED}_cfg${CURRENT_CFG}"
            TARGET_DIR=$(find "$SAMPLES_ROOT" -maxdepth 1 -type d -name "$DIR_PATTERN_PRIMARY" | head -n 1)
            FOUND_TYPE="Primary Match"

            if [ -z "$TARGET_DIR" ] && [ "$VQ_EXPECTED" != "vq-base" ]; then
                DIR_PATTERN_FALLBACK="sample_step${step}_${GPT_TAG}_vq-base_cfg${CURRENT_CFG}"
                TARGET_DIR=$(find "$SAMPLES_ROOT" -maxdepth 1 -type d -name "$DIR_PATTERN_FALLBACK" | head -n 1)
                FOUND_TYPE="Fallback Match (vq-base)"
            fi
        fi

        if [ -z "$TARGET_DIR" ] || [ ! -d "$TARGET_DIR" ]; then
            echo ">>> [Skip] Directory not found for Step $step (CFG $CURRENT_CFG)."
            continue
        fi

        SCORE_FILE="${TARGET_DIR}/score.txt"
        EVALUATED_JUST_NOW=false

        if [ ! -f "$SCORE_FILE" ]; then
            echo ">>> Found Target ($FOUND_TYPE): $(basename "$TARGET_DIR")"
            echo ">>> Running Offline HPSv2 Evaluation..."

            python "$EVAL_SCRIPT" \
                --image_path "$TARGET_DIR" \
                --hpdv2_path "$HPDV2_PATH" \
                --hps_model_path "$HPS_MODEL_PATH" \
                --batch_size 64

            EVALUATED_JUST_NOW=true
            echo ">>> Step $step Evaluation Finished."
        else
            echo ">>> [Check] Already evaluated ($FOUND_TYPE)."
            echo ">>> Path: $(basename "$TARGET_DIR")"
            EVALUATED_JUST_NOW=true
        fi

        if [ -f "$SCORE_FILE" ] && [ "$EVALUATED_JUST_NOW" = true ]; then
            {
                echo "=========================================================="
                echo "[$(date +'%Y-%m-%d %H:%M:%S')] Step: $step | Combo: $GPT_TAG + $VQ_EXPECTED | CFG: $CURRENT_CFG"
                cat "$SCORE_FILE"
                echo ""
            } >> "$CONSOLIDATED_SCORE_FILE"
            echo ">>> Result appended to $CONSOLIDATED_SCORE_FILE"
        fi

        # If this is pure base mode (COMBO_ID=0 or step=base), evaluate once and exit the current CFG loop
        if [ "$COMBO_ID" -eq 0 ] || [ "$step" == "base" ]; then
            if [ -f "$SCORE_FILE" ]; then
                echo ">>> [Notice] Base mode evaluation complete. Skipping remaining steps for CFG $CURRENT_CFG."
                break
            fi
        fi
    done
done

echo ">>> All Done. Consolidated results saved to: ${EXP_ROOT}/results_hpsv2_cfg*.txt"
