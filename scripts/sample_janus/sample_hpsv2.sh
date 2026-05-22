#!/bin/bash
#SBATCH --job-name=ranke_janus_sample_hpsv2
#SBATCH --output=logs/hpsv2_sample_janus_%j.out
#SBATCH --error=logs/hpsv2_sample_janus_%j.err
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
STEPS=(1000 2000 3000 4000 5000 6000)

# --- B. Combination mode selection ---
# In Janus, GPT and VQ are coupled, so only the full-model Online/EMA variants are exposed:
# 0: Base model (JANUS_MODEL_PATH, active when RUN_NAME="")
# 1: janus_finetuned (Online)
# 2: janus_ema       (EMA)
COMBO_ID=1

# --- C. Inference hyperparameters ---
CFG_LIST=(5.0)
BATCH_SIZE=64         # Total images per GPU step
IMAGE_SIZE=384        # Janus default image size: 384

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
SAMPLING_SCRIPT="${CODE_ROOT}/janus/sample/sample_t2i_ddp_hpsv2.py"

export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH}"

# Local HPDv2 directory
HPDV2_PATH="${HPDV2_EVAL_ROOT}"

# Distributed configuration
if [ -n "$SLURM_JOB_ID" ]; then
    echo ">>> [Env] Running via SLURM (sbatch)"
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export NNODES=${SLURM_NNODES:-1}
    export NODE_RANK=${SLURM_NODEID:-0}
    if [ -n "$SLURM_GPUS_ON_NODE" ]; then
        export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
    else
        export GPUS_PER_NODE=8
    fi
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
else
    echo ">>> [Env] Running via direct bash"
    export MASTER_ADDR="127.0.0.1"
    export NNODES=1
    export NODE_RANK=0
    export CUDA_VISIBLE_DEVICES="0,1"
    export GPUS_PER_NODE=2
    echo ">>> [Env] Using $GPUS_PER_NODE GPUs (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
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
echo ">>> Janus HPSv2 Sampling | Combo: $COMBO_TAG | CFG: ${CFG_LIST[*]}"
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

        FINAL_OUTPUT_DIR="${EXP_ROOT}/samples_hpsv2/${OUT_DIR_NAME}"
        SCORE_FILE="${FINAL_OUTPUT_DIR}/score.txt"

        # If the evaluation score already exists, this run is already complete
        if [ -f "$SCORE_FILE" ]; then
            echo ">>> [Skip] Already completely evaluated: $OUT_DIR_NAME"
            continue
        fi

        # Check whether sampling is already complete (the HPSv2 all-category set is about 3200 images)
        if [ -d "$FINAL_OUTPUT_DIR" ] && [ $(find "$FINAL_OUTPUT_DIR" -type f -name "*.jpg" 2>/dev/null | wc -l) -ge 3000 ]; then
            echo ">>> [Notice] Sampling seems complete for $OUT_DIR_NAME, waiting for evaluation."
            continue
        fi

        mkdir -p "$FINAL_OUTPUT_DIR"

        echo ">>> Running HPSv2 Sampling..."
        echo "    Base Model : $BASE_MODEL_PATH"
        echo "    Target Ckpt: $TARGET_CKPT_PATH"
        echo "    Output Dir : $FINAL_OUTPUT_DIR"

        torchrun \
            --nnodes=$NNODES \
            --nproc_per_node=$GPUS_PER_NODE \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            "$SAMPLING_SCRIPT" \
            --base_model_path "$BASE_MODEL_PATH" \
            --target_ckpt_path "$TARGET_CKPT_PATH" \
            --save-dir "$FINAL_OUTPUT_DIR" \
            --hpdv2-path "$HPDV2_PATH" \
            --cfg_scale "$CURRENT_CFG" \
            --batch_size "$BATCH_SIZE" \
            --temperature 1.0

        echo ">>> Step $step sampling finished."
        
        # In pure base mode the weights never change, so sample once and exit the current CFG loop
        if [ "$COMBO_ID" -eq 0 ] || [ "$step" == "base" ]; then
            break
        fi
    done
done

echo ">>> [Job Done] All steps processed."
