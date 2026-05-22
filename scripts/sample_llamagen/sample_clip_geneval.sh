#!/bin/bash
#SBATCH --job-name=ranke_llamagen_sample_geneval
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
# 1. Core configuration (USER CONFIG)
# ==============================================================================

# --- A. Run source (same logic as the public training scripts) ---
# To test RL results, set the RL run name
# To test SFT results, set the SFT run name (the root directory will switch automatically)
RUN_NAME=""   # Fill in the run name produced by training; leave empty to use base-model logic

# Checkpoint steps to sample
STEPS=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000)

# --- B. Combination mode selection ---
# 1: GPT(Online) + VQ(Online)
# 2: GPT(Online) + VQ(EMA)
# 3: GPT(EMA)    + VQ(Online)
# 4: GPT(EMA)    + VQ(EMA)
# 0: GPT(Base) + VQ(Base) -> active only when RUN_NAME=""
COMBO_ID=1

# --- C. Inference hyperparameters ---
IMAGE_SIZE=256
CFG_SCALE=6.0      # The training setup often uses 1.0 here
BATCH_SIZE=128      # Batch size per GPU

# ==============================================================================
# 2. Infrastructure and path handling
# ==============================================================================
# Load the shared environment config
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

# Automatically detect task type and set the root directory
TASK_ROOT="${PROJECT_OUTPUT_ROOT}/ranke_llamagen"

EXP_ROOT="${TASK_ROOT}/${RUN_NAME}"
SAMPLING_SCRIPT="${CODE_ROOT}/autoregressive/sample/sample_t2i_ddp_geneval.py"
PROMPTS_FILE="${GENEVAL_PROMPTS_FILE:-${CODE_ROOT}/evaluations/geneval/prompts/evaluation_metadata.jsonl}"
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH}"

# Distributed runtime configuration
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
GPUS=8

# Resolve checkpoint filename mappings
case "$COMBO_ID" in
    0) GPT_T="base";                VQ_T="base";            GPT_TAG="gpt-base"; VQ_TAG="vq-base" ;;
    1) GPT_T="gpt_finetuned.pt";    VQ_T="vq_finetuned.pt"; GPT_TAG="gpt-onl";  VQ_TAG="vq-onl"  ;;
    2) GPT_T="gpt_finetuned.pt";    VQ_T="vq_ema.pt";       GPT_TAG="gpt-onl";  VQ_TAG="vq-ema"  ;;
    3) GPT_T="gpt_ema.pt";          VQ_T="vq_finetuned.pt"; GPT_TAG="gpt-ema";  VQ_TAG="vq-onl"  ;;
    4) GPT_T="gpt_ema.pt";          VQ_T="vq_ema.pt";       GPT_TAG="gpt-ema";  VQ_TAG="vq-ema"  ;;
    *) echo "Error: Invalid COMBO_ID"; exit 1 ;;
esac

# ==============================================================================
# 3. Sampling loop
# ==============================================================================

# If RUN_NAME is empty, force a single base-model sampling run
if [ -z "$RUN_NAME" ]; then
    loop_steps=("base")
    echo ">>> [Mode] RUN_NAME is empty. Sampling BASE model."
else
    loop_steps=("${STEPS[@]}")
fi

for step in "${loop_steps[@]}"; do
    echo ">>> ---------------------------------------------------"
    
    if [ "$step" == "base" ]; then
        # Load the stage-1 base model (256 resolution)
        GPT_PATH="$GPT_CKPT_PATH_STAGE1"
        FINAL_VQ_PATH="$VQ_CKPT_PATH"
        OUT_DIR_NAME="baseline_gpt-base_vq-base_cfg${CFG_SCALE}"
    else
        echo ">>> Processing Checkpoint Step: $step"
        CKPT_DIR="${EXP_ROOT}/checkpoint_${step}"
        
        if [ ! -d "$CKPT_DIR" ]; then
            echo ">>> [Skip] Directory not found: $CKPT_DIR"
            continue
        fi

        GPT_PATH="${CKPT_DIR}/${GPT_T}"
        # VQ priority: fine-tuned weights first, then base weights
        CANDIDATE_VQ="${CKPT_DIR}/${VQ_T}"
        if [ -f "$CANDIDATE_VQ" ]; then
            FINAL_VQ_PATH="$CANDIDATE_VQ"
            ACTUAL_VQ_TAG="$VQ_TAG"
        else
            FINAL_VQ_PATH="$VQ_CKPT_PATH"
            ACTUAL_VQ_TAG="vq-base"
        fi
        OUT_DIR_NAME="sample_step${step}_${GPT_TAG}_${ACTUAL_VQ_TAG}_cfg${CFG_SCALE}"
    fi

    # File checks
    if [ ! -f "$GPT_PATH" ]; then
        echo ">>> [Error] GPT not found: $GPT_PATH"
        continue
    fi

    FINAL_OUTPUT_DIR="${EXP_ROOT}/samples_geneval/${OUT_DIR_NAME}"
    mkdir -p "$FINAL_OUTPUT_DIR"

    echo ">>> Running Torchrun..."
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
done

echo ">>> [Job Done] All steps processed."