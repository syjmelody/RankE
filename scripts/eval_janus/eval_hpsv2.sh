#!/bin/bash
#SBATCH --job-name=ranke_janus_eval_hpsv2
#SBATCH --output=logs/hpsv2_eval_janus_%j.out
#SBATCH --error=logs/hpsv2_eval_janus_%j.err
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
# [1] Core configuration (must match the sampling script)
# ==============================================================================

# --- A. Run source ---
# Set the RL training run name; leave empty to use the base model
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

EVAL_SCRIPT="${LOCAL_HPSV2_REPO}/eval_hpsv2.py"
TASK_ROOT="${PROJECT_OUTPUT_ROOT}/ranke_janus"

if [ -z "$RUN_NAME" ]; then
    EXP_ROOT="${TASK_ROOT}/base_model_evaluation"
else
    EXP_ROOT="${TASK_ROOT}/${RUN_NAME}"
fi

SAMPLES_ROOT="${EXP_ROOT}/samples_hpsv2"

case "$COMBO_ID" in
    0) COMBO_TAG="janus-base" ;;
    1) COMBO_TAG="janus-onl"  ;;
    2) COMBO_TAG="janus-ema"  ;;
    *) echo "Error: Invalid COMBO_ID $COMBO_ID"; exit 1 ;;
esac

echo "========================================================================"
echo ">>> Offline HPSv2 Evaluation Pipeline (Janus) | Combo Key: $COMBO_ID ($COMBO_TAG)"
echo ">>> CFG List   : ${CFG_LIST[*]}"
echo ">>> Steps      : ${STEPS[*]}"
echo "========================================================================"

# ==============================================================================
# [3] Evaluation loop
# ==============================================================================

if [ -z "$RUN_NAME" ]; then
    loop_steps=("base")
else
    loop_steps=("${STEPS[@]}")
fi

# Outer loop over CFG values
for CURRENT_CFG in "${CFG_LIST[@]}"; do
    
    CONSOLIDATED_SCORE_FILE="${EXP_ROOT}/results_hpsv2_cfg${CURRENT_CFG}.txt"

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
                echo "[$(date +'%Y-%m-%d %H:%M:%S')] Step: $step | Combo: $COMBO_TAG | CFG: $CURRENT_CFG"
                cat "$SCORE_FILE"
                echo ""
            } >> "$CONSOLIDATED_SCORE_FILE"
            echo ">>> Result appended to $CONSOLIDATED_SCORE_FILE"
        fi

        # In pure base mode, evaluate once and exit the current CFG loop
        if [ "$COMBO_ID" -eq 0 ] || [ "$step" == "base" ]; then
            if [ -f "$SCORE_FILE" ]; then
                echo ">>> [Notice] Base mode evaluation complete. Skipping remaining steps for CFG $CURRENT_CFG."
                break
            fi
        fi
    done
done

echo ">>> All Done. Consolidated results saved to: ${EXP_ROOT}/results_hpsv2_cfg*.txt"
