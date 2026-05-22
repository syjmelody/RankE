#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=logs/coco_sample_%j.out
#SBATCH --error=logs/coco_sample_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256GB
#SBATCH --time=24:00:00   # COCO 30k-image sampling can take a while; leave enough time

set -e
export PYTHONUNBUFFERED=1

# ==============================================================================
# [1] Core configuration (USER CONFIG)
# ==============================================================================

# --- A. Run source ---
# Fill in the run name produced by training
# Leave empty ("") to test the base pretrained model
RUN_NAME=""   # Fill in the run name produced by training; leave empty to use base-model logic

# Checkpoint steps to sample
STEPS=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000)

# --- B. Sampling parameters (supports nested loops) ---
CFG_LIST=(6.0)

# --- C. Combination mode selection ---
# 1: GPT(Online) + VQ(Online)
# 2: GPT(Online) + VQ(EMA)
# 3: GPT(EMA)    + VQ(Online)
# 4: GPT(EMA)    + VQ(EMA)
# 5: GPT(Online) + VQ(Base)   <--- policy-only test mode; keep the original VQ fixed
# 0: force pure base mode (automatically used when RUN_NAME="")
COMBO_ID=1

# --- D. Hardware and sample-count configuration ---
IMAGE_SIZE=256
PER_GPU_BATCH=128     # At 256 resolution, 64-128 usually works; reduce this if you hit OOM
NUM_SAMPLES=30000    # Standard COCO sample count for FID

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

# Use the public LlamaGen output root
TASK_ROOT="${PROJECT_OUTPUT_ROOT}/ranke_llamagen"

# Automatically detect task type and set the experiment root directory
if [ -z "$RUN_NAME" ]; then
    EXP_ROOT="${TASK_ROOT}/base_model_evaluation"
else
    EXP_ROOT="${TASK_ROOT}/${RUN_NAME}"
fi

# Core entry point: COCO sampling uses sample_t2i_ddp.py
SAMPLING_SCRIPT="${CODE_ROOT}/autoregressive/sample/sample_t2i_ddp.py"
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH}"

# Distributed runtime configuration
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export GPUS_PER_NODE=8

# Resolve the combo mode
case "$COMBO_ID" in
    0) GPT_T="base";             VQ_T="base";            GPT_TAG="gpt-base"; VQ_TAG="vq-base" ;;
    1) GPT_T="gpt_finetuned.pt"; VQ_T="vq_finetuned.pt"; GPT_TAG="gpt-onl";  VQ_TAG="vq-onl"  ;;
    2) GPT_T="gpt_finetuned.pt"; VQ_T="vq_ema.pt";       GPT_TAG="gpt-onl";  VQ_TAG="vq-ema"  ;;
    3) GPT_T="gpt_ema.pt";       VQ_T="vq_finetuned.pt"; GPT_TAG="gpt-ema";  VQ_TAG="vq-onl"  ;;
    4) GPT_T="gpt_ema.pt";       VQ_T="vq_ema.pt";       GPT_TAG="gpt-ema";  VQ_TAG="vq-ema"  ;;
    # Mode 5: use the fine-tuned GPT while keeping the base VQ weights
    5) GPT_T="gpt_finetuned.pt"; VQ_T="base";            GPT_TAG="gpt-onl";  VQ_TAG="vq-base" ;;
    *) echo "Error: Invalid COMBO_ID"; exit 1 ;;
esac

echo "========================================================================"
echo ">>> Multi-Step & CFG Sweep COCO Sampling Started"
echo ">>> Run Name  : ${RUN_NAME:-BASE_MODEL}"
echo ">>> Exp Root  : ${EXP_ROOT}"
echo ">>> Steps     : ${STEPS[*]}"
echo ">>> CFG List  : ${CFG_LIST[*]}"
echo ">>> Combo     : $COMBO_ID ($GPT_TAG + $VQ_TAG)"
echo "========================================================================"

# ==============================================================================
# [3] Two-level sampling loop (outer: CFG, inner: step)
# ==============================================================================

# In base mode, force steps=base and run only once
if [ -z "$RUN_NAME" ]; then
    loop_steps=("base")
else
    loop_steps=("${STEPS[@]}")
fi

for CURRENT_CFG in "${CFG_LIST[@]}"; do
    for step in "${loop_steps[@]}"; do
        
        echo ">>> ---------------------------------------------------"
        
        # A. Path-resolution logic
        if [ "$step" == "base" ]; then
            echo ">>> Processing Baseline Model | CFG: $CURRENT_CFG"
            GPT_PATH="$GPT_CKPT_PATH_STAGE1"
            FINAL_VQ_PATH="$VQ_CKPT_PATH"
            ACTUAL_VQ_TAG="vq-base"
            OUT_DIR_NAME="baseline_gpt-base_vq-base_cfg${CURRENT_CFG}"
        else
            echo ">>> Processing Checkpoint: Step $step | CFG: $CURRENT_CFG"
            CKPT_DIR="${EXP_ROOT}/checkpoint_${step}"
            
            if [ ! -d "$CKPT_DIR" ]; then
                echo ">>> [Skip] Checkpoint dir not found: $CKPT_DIR"
                continue
            fi

            GPT_PATH="${CKPT_DIR}/${GPT_T}"
            
            # VQ fallback logic plus forced-base handling
            if [ "$VQ_T" == "base" ]; then
                # For COMBO_ID=5, force the base VQ from the global config
                FINAL_VQ_PATH="$VQ_CKPT_PATH"
                ACTUAL_VQ_TAG="vq-base"
            else
                CANDIDATE_VQ="${CKPT_DIR}/${VQ_T}"
                if [ -f "$CANDIDATE_VQ" ]; then
                    FINAL_VQ_PATH="$CANDIDATE_VQ"
                    ACTUAL_VQ_TAG="$VQ_TAG"
                else
                    echo ">>> [Warning] VQ $VQ_T not found. Fallback to vq-base."
                    FINAL_VQ_PATH="$VQ_CKPT_PATH"
                    ACTUAL_VQ_TAG="vq-base"
                fi
            fi
            
            OUT_DIR_NAME="sample_step${step}_${GPT_TAG}_${ACTUAL_VQ_TAG}_cfg${CURRENT_CFG}"
        fi

        # Validate the GPT checkpoint path
        if [ ! -f "$GPT_PATH" ]; then
            echo ">>> [Error] GPT model not found at $GPT_PATH"
            continue
        fi

        SAMPLE_OUT_DIR="${EXP_ROOT}/samples_coco/${OUT_DIR_NAME}"
        
        # [Skip-if-done] Skip when the target directory already contains NUM_SAMPLES images
        if [ -d "$SAMPLE_OUT_DIR/images" ]; then
            IMG_COUNT=$(find "$SAMPLE_OUT_DIR/images" -type f -name "*.png" | wc -l)
            if [ "$IMG_COUNT" -ge "$NUM_SAMPLES" ]; then
                 echo ">>> [Skip] Already generated $IMG_COUNT images in $OUT_DIR_NAME"
                 continue
            fi
        fi

        mkdir -p "$SAMPLE_OUT_DIR"
        echo "    GPT: $GPT_PATH"
        echo "    VQ:  $FINAL_VQ_PATH"
        echo "    Out: $SAMPLE_OUT_DIR"

        # B. Launch sampling
        torchrun \
            --nnodes=1 \
            --nproc_per_node=$GPUS_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            "$SAMPLING_SCRIPT" \
            --gpt-ckpt "$GPT_PATH" \
            --vq-ckpt "$FINAL_VQ_PATH" \
            --t5-path "$T5_PATH" \
            --prompt-csv "$EVAL_PROMPT_FILE" \
            --sample-dir "$SAMPLE_OUT_DIR" \
            --gpt-model "GPT-XL" \
            --image-size "$IMAGE_SIZE" \
            --cfg-scale "$CURRENT_CFG" \
            --per-proc-batch-size "$PER_GPU_BATCH" \
            --num-fid-samples "$NUM_SAMPLES" \
            --precision "bf16"

        echo ">>> Finished Step: $step | CFG: $CURRENT_CFG"

    done
done

echo "========================================================================"
echo ">>> All COCO Sampling Sweeps Completed Successfully."
echo "========================================================================"