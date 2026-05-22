#!/bin/bash
#SBATCH --job-name=ranke_llamagen_eval_hps_geneval
#SBATCH --output=logs/geneval_eval_%j.out
#SBATCH --error=logs/geneval_eval_%j.err
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
# [1] Core configuration (USER CONFIG)
# ==============================================================================
RUN_NAME=""   # Fill in a run name that has already been sampled; leave empty to use base-model logic
STEPS=(500 1000 1500 2000 2500 3000)
COMBO_ID=2
CFG_SCALE=6.0

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
    0) GPT_TAG="gpt-base"; VQ_EXPECTED="vq-base" ;;
    1) GPT_TAG="gpt-onl";  VQ_EXPECTED="vq-onl"  ;;
    2) GPT_TAG="gpt-onl";  VQ_EXPECTED="vq-ema"  ;;
    3) GPT_TAG="gpt-ema";  VQ_EXPECTED="vq-onl"  ;;
    4) GPT_TAG="gpt-ema";  VQ_EXPECTED="vq-ema"  ;;
    5) GPT_TAG="gpt-onl";  VQ_EXPECTED="vq-base" ;;
    *) echo "Error: Invalid COMBO_ID $COMBO_ID"; exit 1 ;;
esac

TASK_ROOT="${PROJECT_OUTPUT_ROOT}/ranke_llamagen"

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
echo ">>> GenEval Evaluation Pipeline (Both mode, HPSv2 reward)"
echo ">>> Combo Key  : $COMBO_ID ($GPT_TAG + $VQ_EXPECTED)"
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
        TARGET_DIR="${SAMPLES_ROOT}/baseline_gpt-base_vq-base_cfg${CFG_SCALE}"
        FOUND_TYPE="Base Model"
    else
        DIR_PATTERN_PRIMARY="sample_step${step}_${GPT_TAG}_${VQ_EXPECTED}_cfg${CFG_SCALE}"
        TARGET_DIR=$(find "$SAMPLES_ROOT" -maxdepth 1 -type d -name "$DIR_PATTERN_PRIMARY" | head -n 1)
        FOUND_TYPE="Primary Match"

        if [ -z "$TARGET_DIR" ] && [ "$VQ_EXPECTED" != "vq-base" ]; then
            DIR_PATTERN_FALLBACK="sample_step${step}_${GPT_TAG}_vq-base_cfg${CFG_SCALE}"
            TARGET_DIR=$(find "$SAMPLES_ROOT" -maxdepth 1 -type d -name "$DIR_PATTERN_FALLBACK" | head -n 1)
            FOUND_TYPE="Fallback Match (vq-base)"
        fi
    fi

    if [ -z "$TARGET_DIR" ] || [ ! -d "$TARGET_DIR" ]; then
        echo ">>> [Skip] Directory not found for Step $step (CFG $CFG_SCALE)."
        continue
    fi

    RESULTS_JSONL="${TARGET_DIR}/results.jsonl"
    FINAL_SCORE="${TARGET_DIR}/final_score.txt"
    EVALUATED_JUST_NOW=false

    if [ ! -f "$FINAL_SCORE" ]; then
        echo ">>> Found Target ($FOUND_TYPE): $(basename "$TARGET_DIR")"
        echo ">>> Running GenEval Evaluation..."

        if [ ! -f "$RESULTS_JSONL" ] || [ ! -s "$RESULTS_JSONL" ]; then
            python "$EVAL_SCRIPT" "$TARGET_DIR" \
                --outfile "$RESULTS_JSONL" \
                --model-path "$MODEL_PATH"
        else
            echo ">>> [Notice] results.jsonl exists, skipping detection step."
        fi

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

    if [ -f "$FINAL_SCORE" ] && [ "$EVALUATED_JUST_NOW" = true ]; then
        echo ">>> Result for Step $step:"
        cat "$FINAL_SCORE"

        {
            echo "=========================================================="
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Step: $step | Combo: $GPT_TAG + $VQ_EXPECTED | CFG: $CFG_SCALE"
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
