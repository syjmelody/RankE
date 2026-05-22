#!/bin/bash
#SBATCH --job-name=ranke_llamagen_sample_hps_geneval
#SBATCH --output=logs/geneval_sample_%j.out
#SBATCH --error=logs/geneval_sample_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256GB
#SBATCH --time=12:00:00

set -e
export PYTHONUNBUFFERED=1

# ==============================================================================
# [1] Core configuration
# ==============================================================================
RUN_NAME=""   # Fill in the run name produced by training; leave empty to use base-model logic
STEPS=(500 1000 1500 2000 2500 3000)
COMBO_ID=2
IMAGE_SIZE=256
CFG_SCALE=6.0
BATCH_SIZE=128

# ==============================================================================
# [2] Environment detection (bash vs sbatch)
# ==============================================================================
if [ -n "$SLURM_JOB_ID" ]; then
    echo ">>> [Env] Running via SLURM (sbatch)"
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

    if [ -n "$SLURM_GPUS_ON_NODE" ]; then
        GPUS=$SLURM_GPUS_ON_NODE
    else
        GPUS=8
    fi
    echo ">>> [Env] Using $GPUS GPUs on node"
else
    echo ">>> [Env] Running via direct bash"
    export MASTER_ADDR="127.0.0.1"

    export CUDA_VISIBLE_DEVICES="0,1"
    GPUS=2

    echo ">>> [Env] Using $GPUS GPUs (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
fi
export MASTER_PORT=$(python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# ==============================================================================
# [3] Path and model-loading logic
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
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH}"

SAMPLING_SCRIPT="${CODE_ROOT}/autoregressive/sample/sample_t2i_ddp_geneval.py"
PROMPTS_FILE="${GENEVAL_PROMPTS_FILE:-${CODE_ROOT}/evaluations/geneval/prompts/evaluation_metadata.jsonl}"
TASK_ROOT="${PROJECT_OUTPUT_ROOT}/ranke_llamagen"
EXP_ROOT="${TASK_ROOT}/${RUN_NAME:-base_model_evaluation}"
SAMPLES_ROOT="${EXP_ROOT}/samples_geneval"

case "$COMBO_ID" in
    0) GPT_T="base";             VQ_T="base";            GPT_TAG="gpt-base"; VQ_TAG="vq-base" ;;
    1) GPT_T="gpt_finetuned.pt"; VQ_T="vq_finetuned.pt"; GPT_TAG="gpt-onl";  VQ_TAG="vq-onl"  ;;
    2) GPT_T="gpt_finetuned.pt"; VQ_T="vq_ema.pt";       GPT_TAG="gpt-onl";  VQ_TAG="vq-ema"  ;;
    3) GPT_T="gpt_ema.pt";       VQ_T="vq_finetuned.pt"; GPT_TAG="gpt-ema";  VQ_TAG="vq-onl"  ;;
    4) GPT_T="gpt_ema.pt";       VQ_T="vq_ema.pt";       GPT_TAG="gpt-ema";  VQ_TAG="vq-ema"  ;;
    5) GPT_T="gpt_finetuned.pt"; VQ_T="base";            GPT_TAG="gpt-onl";  VQ_TAG="vq-base" ;;
    *) echo "Error: Invalid COMBO_ID"; exit 1 ;;
esac

echo "========================================================================"
echo ">>> GenEval Sampling Started (Both mode, HPSv2 reward)"
echo ">>> Combo: $COMBO_ID ($GPT_TAG + $VQ_TAG)"
echo ">>> GPUs : $GPUS"
echo "========================================================================"

loop_steps=("${STEPS[@]}")
if [ "$COMBO_ID" -eq 0 ]; then
    loop_steps=("base")
elif [ -z "$RUN_NAME" ]; then
    loop_steps=("base")
fi

# ==============================================================================
# [4] Main loop: sampling
# ==============================================================================
for step in "${loop_steps[@]}"; do
    echo ">>> ---------------------------------------------------"

    if [ "$step" == "base" ]; then
        GPT_PATH="$GPT_CKPT_PATH_STAGE1"
        FINAL_VQ_PATH="$VQ_CKPT_PATH"
        ACTUAL_VQ_TAG="vq-base"
        OUT_DIR_NAME="baseline_gpt-base_vq-base_cfg${CFG_SCALE}"
    else
        CKPT_DIR="${EXP_ROOT}/checkpoint_${step}"
        [ ! -d "$CKPT_DIR" ] && echo ">>> [Skip] Not found: $CKPT_DIR" && continue

        GPT_PATH="${CKPT_DIR}/${GPT_T}"
        if [ "$VQ_T" == "base" ]; then
            FINAL_VQ_PATH="$VQ_CKPT_PATH"
            ACTUAL_VQ_TAG="vq-base"
        else
            if [ -f "${CKPT_DIR}/${VQ_T}" ]; then
                FINAL_VQ_PATH="${CKPT_DIR}/${VQ_T}"
                ACTUAL_VQ_TAG="$VQ_TAG"
            else
                FINAL_VQ_PATH="$VQ_CKPT_PATH"
                ACTUAL_VQ_TAG="vq-base"
            fi
        fi
        OUT_DIR_NAME="sample_step${step}_${GPT_TAG}_${ACTUAL_VQ_TAG}_cfg${CFG_SCALE}"
    fi

    [ ! -f "$GPT_PATH" ] && echo ">>> [Error] GPT not found: $GPT_PATH" && continue

    FINAL_OUTPUT_DIR="${SAMPLES_ROOT}/${OUT_DIR_NAME}"
    mkdir -p "$FINAL_OUTPUT_DIR"

    echo ">>> Running GenEval Sampling for Step: $step | CFG: $CFG_SCALE"
    echo "    GPT: $GPT_PATH"
    echo "    VQ:  $FINAL_VQ_PATH"
    echo "    Out: $FINAL_OUTPUT_DIR"

    torchrun --nnodes=1 --nproc_per_node=$GPUS \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        "$SAMPLING_SCRIPT" \
        --prompts "$PROMPTS_FILE" \
        --save-dir "$FINAL_OUTPUT_DIR" \
        --gpt-ckpt "$GPT_PATH" \
        --vq-ckpt "$FINAL_VQ_PATH" \
        --t5-path "$T5_PATH" \
        --gpt-model "GPT-XL" \
        --image-size "$IMAGE_SIZE" \
        --cfg-scale "$CFG_SCALE" \
        --repeat 4 \
        --batch-size "$BATCH_SIZE" \
        --precision "bf16"

    echo ">>> Step $step sampling finished."
    [ "$step" == "base" ] && break
done

echo ">>> All GenEval Sampling Completed."
