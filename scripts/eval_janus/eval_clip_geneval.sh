#!/bin/bash
#SBATCH --job-name=ranke_janus_eval_geneval
#SBATCH --output=logs/geneval_eval_janus_%j.out
#SBATCH --error=logs/geneval_eval_janus_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=12:00:00

set -e
export PYTHONUNBUFFERED=1

# ==============================================================================
# [1] Core configuration (must match the sampling script)
# ==============================================================================

# --- A. Run source ---
RUN_NAME=""   # Fill in a run name that has already been sampled; leave empty to use base-model logic

# Checkpoint steps to iterate over (space-separated)
STEPS=(1000 2000 3000 4000 5000 6000)

# --- B. Combination mode selection (aligned with the sampling script) ---
# 0: Base model
# 1: janus_finetuned (Online)
# 2: janus_ema       (EMA)
COMBO_ID=1

# --- C. Inference configuration ---
CFG_SCALE=5.0

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

SAMPLES_ROOT="${EXP_ROOT}/samples_geneval"
GENEVAL_ROOT="${CODE_ROOT}/evaluations/geneval"
EVAL_SCRIPT="${GENEVAL_ROOT}/evaluation/evaluate_images.py"
SUMMARY_SCRIPT="${GENEVAL_ROOT}/evaluation/summary_scores.py"
MODEL_PATH="${GENEVAL_MASK2FORMER_PATH:-${STORAGE_ROOT}/ckpt_hf/tsbpp/geneval_mask2former}"

if [ ! -d "$MODEL_PATH" ]; then
    echo ">>> [Error] GenEval Mask2Former model not found at: $MODEL_PATH"
    exit 1
fi

CONSOLIDATED_SCORE_FILE="${EXP_ROOT}/results_geneval_cfg${CFG_SCALE}.txt"

export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH}"
mkdir -p "$EXP_ROOT"

echo "========================================================================"
echo ">>> GenEval Evaluation Pipeline (Janus) | Combo: $COMBO_TAG"
echo ">>> CFG Scale  : $CFG_SCALE"
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

for step in "${loop_steps[@]}"; do

    echo ">>> ---------------------------------------------------"
    echo ">>> Processing Step: $step"

    if [ "$step" == "base" ]; then
        TARGET_DIR="${SAMPLES_ROOT}/baseline_janus-base_cfg${CFG_SCALE}"
        FOUND_TYPE="Base Model"
    else
        DIR_PATTERN="sample_step${step}_${COMBO_TAG}_cfg${CFG_SCALE}"
        TARGET_DIR="${SAMPLES_ROOT}/${DIR_PATTERN}"
        FOUND_TYPE="Primary Match"
    fi

    if [ ! -d "$TARGET_DIR" ]; then
        echo ">>> [Skip] Directory not found: $(basename "$TARGET_DIR")"
        continue
    fi

    RESULTS_JSONL="${TARGET_DIR}/results.jsonl"
    FINAL_SCORE="${TARGET_DIR}/final_score.txt"
    EVALUATED_JUST_NOW=false

    if [ ! -f "$FINAL_SCORE" ]; then
        echo ">>> Found Target ($FOUND_TYPE): $(basename "$TARGET_DIR")"
        echo ">>> Running GenEval Evaluation..."

        # Step A: Object detection & scoring
        if [ ! -f "$RESULTS_JSONL" ] || [ ! -s "$RESULTS_JSONL" ]; then
            python "$EVAL_SCRIPT" "$TARGET_DIR" \
                --outfile "$RESULTS_JSONL" \
                --model-path "$MODEL_PATH"
        else
            echo ">>> [Notice] results.jsonl exists, skipping detection step."
        fi

        # Step B: Score summary
        if [ -f "$RESULTS_JSONL" ]; then
            python "$SUMMARY_SCRIPT" "$RESULTS_JSONL" > "$FINAL_SCORE"
            EVALUATED_JUST_NOW=true
            echo ">>> Step $step Evaluation Finished."
        fi
    else
        echo ">>> [Check] Already evaluated ($FOUND_TYPE)."
        echo ">>> Path: $(basename "$TARGET_DIR")"
        EVALUATED_JUST_NOW=true
    fi

    # Result display & consolidated log
    if [ -f "$FINAL_SCORE" ] && [ "$EVALUATED_JUST_NOW" = true ]; then
        echo ">>> Result for Step $step:"
        cat "$FINAL_SCORE"

        {
            echo "=========================================================="
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Step: $step | Combo: $COMBO_TAG | CFG: $CFG_SCALE"
            echo "Dir: $(basename "$TARGET_DIR")"
            echo "----------------------------------------------------------"
            cat "$FINAL_SCORE"
            echo ""
        } >> "$CONSOLIDATED_SCORE_FILE"

        echo ">>> [Success] Result appended to $CONSOLIDATED_SCORE_FILE"
    elif [ ! -f "$FINAL_SCORE" ]; then
        echo ">>> [Warning] Evaluation completed but $FINAL_SCORE was not found!"
    fi

    if [ "$step" == "base" ] && [ -f "$FINAL_SCORE" ]; then
        echo ">>> [Notice] Base mode evaluation complete."
        break
    fi

done

echo ">>> ---------------------------------------------------"
echo ">>> All Done. Consolidated results saved to:"
echo ">>> $CONSOLIDATED_SCORE_FILE"
