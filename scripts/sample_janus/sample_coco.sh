#!/bin/bash
#SBATCH --job-name=ranke_janus_sample_coco
#SBATCH --output=logs/coco_sample_janus_%j.out
#SBATCH --error=logs/coco_sample_janus_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512GB
#SBATCH --time=12:00:00

set -e
export PYTHONUNBUFFERED=1

# ==============================================================================
# [1] Core configuration (USER CONFIG)
# ==============================================================================

# --- A. Run source ---
# Set the RL training run name; leave empty to use the base model
RUN_NAME=""   # Fill in the run name produced by training; leave empty to use base-model logic

# Checkpoint steps to iterate over (space-separated)
# STEPS=(500 1000 1500 2000 2500 3000 3500 4000)
STEPS=(4500 5000 5500 6000)

# --- B. Combination mode selection ---
# In Janus, GPT and VQ are coupled, so only the full-model Online/EMA variants are exposed:
# 0: Base model (JANUS_MODEL_PATH, active when RUN_NAME="")
# 1: janus_finetuned (Online)
# 2: janus_ema       (EMA)
COMBO_ID=1

# --- C. Inference hyperparameters ---
CFG_LIST=(5.0)
IMAGE_SIZE=384
IMAGE_TOKEN_SIZE=576
PER_GPU_BATCH=64      # Reduce this if you hit OOM (for example 8 or 16)
NUM_SAMPLES=30000     # COCO validation usually samples 30k images for FID

# ==============================================================================
# [2] Infrastructure and path handling
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

TASK_ROOT="${PROJECT_OUTPUT_ROOT}/ranke_janus"
EXP_ROOT="${TASK_ROOT}/${RUN_NAME:-base_model_evaluation}"
SCRIPT_PATH="${CODE_ROOT}/janus/sample/sample_t2i_ddp_coco.py"

export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH}"

# Distributed configuration
if [ -n "$SLURM_JOB_ID" ]; then
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export NNODES=${SLURM_NNODES:-1}
    export NODE_RANK=${SLURM_NODEID:-0}
    export GPUS_PER_NODE=8
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
else
    export MASTER_ADDR="127.0.0.1"
    export NNODES=1
    export NODE_RANK=0
    export GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
fi
export MASTER_PORT=$(python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# Resolve combination tag
case "$COMBO_ID" in
    0) CKPT_SUBDIR="base";            COMBO_TAG="janus-base" ;;
    1) CKPT_SUBDIR="janus_finetuned"; COMBO_TAG="janus-onl"  ;;
    2) CKPT_SUBDIR="janus_ema";       COMBO_TAG="janus-ema"  ;;
    *) echo "Error: Invalid COMBO_ID $COMBO_ID"; exit 1 ;;
esac

# ==============================================================================
# [3] Sampling loop
# ==============================================================================

if [ -z "$RUN_NAME" ]; then
    loop_steps=("base")
    echo ">>> [Mode] RUN_NAME is empty. Sampling BASE model."
else
    loop_steps=("${STEPS[@]}")
fi

echo "========================================================================"
echo ">>> Janus COCO Sampling | Combo: $COMBO_TAG | CFG: ${CFG_LIST[*]}"
echo ">>> Root: $EXP_ROOT"
echo "========================================================================"

for CURRENT_CFG in "${CFG_LIST[@]}"; do
    for step in "${loop_steps[@]}"; do
        echo ">>> ---------------------------------------------------"
        echo ">>> Processing Step: $step | CFG: $CURRENT_CFG"

        BASE_MODEL_PATH="$JANUS_MODEL_PATH"

        if [ "$step" == "base" ]; then
            TARGET_CKPT_PATH="$JANUS_MODEL_PATH"
            OUT_DIR_NAME="baseline_janus-base_cfg${CURRENT_CFG}"
        else
            echo ">>> Processing Checkpoint Step: $step"
            CKPT_DIR="${EXP_ROOT}/checkpoint_${step}"

            if [ ! -d "$CKPT_DIR" ]; then
                echo ">>> [Skip] Directory not found: $CKPT_DIR"
                continue
            fi

            TARGET_CKPT_PATH="${CKPT_DIR}/${CKPT_SUBDIR}"
            if [ ! -d "$TARGET_CKPT_PATH" ]; then
                echo ">>> [Skip] ${CKPT_SUBDIR} not found in checkpoint_${step}"
                continue
            fi
            OUT_DIR_NAME="sample_step${step}_${COMBO_TAG}_cfg${CURRENT_CFG}"
        fi

        SAMPLE_OUT_DIR="${EXP_ROOT}/samples_coco/${OUT_DIR_NAME}"

        # --- Skip logic (avoid duplicate sampling) ---
        if [ -d "$SAMPLE_OUT_DIR/images" ] && [ "$(ls -A "$SAMPLE_OUT_DIR/images" 2>/dev/null)" ]; then
            CHECK_COUNT=$(ls -1 "$SAMPLE_OUT_DIR/images" | wc -l)
            if [ "$CHECK_COUNT" -ge "$NUM_SAMPLES" ]; then
                 echo ">>> [Skip] Already sampled ($CHECK_COUNT images) in: $OUT_DIR_NAME"
                 continue
            fi
        fi

        echo "    Base Model : $BASE_MODEL_PATH"
        echo "    Target Ckpt: $TARGET_CKPT_PATH"
        echo "    Output Dir : $SAMPLE_OUT_DIR"

        # --- Execution ---
        torchrun \
            --nnodes=$NNODES \
            --nproc_per_node=$GPUS_PER_NODE \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            "$SCRIPT_PATH" \
            --base_model_path "$BASE_MODEL_PATH" \
            --target_ckpt_path "$TARGET_CKPT_PATH" \
            --prompt_file "$EVAL_PROMPT_FILE_COCO" \
            --output_dir "$SAMPLE_OUT_DIR" \
            --num_fid_samples "$NUM_SAMPLES" \
            --batch_size "$PER_GPU_BATCH" \
            --cfg_scale "$CURRENT_CFG" \
            --temperature 1.0 \
            --image_token_size "$IMAGE_TOKEN_SIZE"

        echo ">>> Finished Step $step"
        
    done
done

echo ">>> All COCO Sampling Tasks Completed."
