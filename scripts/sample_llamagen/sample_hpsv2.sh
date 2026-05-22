#!/bin/bash
#SBATCH --job-name=ranke_llamagen_sample_hpsv2
#SBATCH --output=logs/hpsv2_sample_%j.out
#SBATCH --error=logs/hpsv2_sample_%j.err
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
STEPS=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000)
CFG_LIST=(6.0)
COMBO_ID=2
IMAGE_SIZE=256
BATCH_SIZE=128

# ==============================================================================
# [2] Environment detection (bash vs sbatch)
# ==============================================================================
if [ -n "$SLURM_JOB_ID" ]; then
    echo ">>> [Env] Running via SLURM (sbatch)"
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    
    # Under Slurm, read the allocated GPU count directly; default to 8 otherwise
    if [ -n "$SLURM_GPUS_ON_NODE" ]; then
        GPUS=$SLURM_GPUS_ON_NODE
    else
        GPUS=8
    fi
    echo ">>> [Env] Using $GPUS GPUs on node"
else
    echo ">>> [Env] Running via direct bash"
    export MASTER_ADDR="127.0.0.1"
    
    # Force binding to GPUs 0 and 1 and hardcode two processes instead of using nvidia-smi
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

# Local HPDv2 directory
HPDV2_PATH="${HPDV2_EVAL_ROOT}"

SAMPLING_SCRIPT="${CODE_ROOT}/autoregressive/sample/sample_t2i_ddp_hpsv2.py"
TASK_ROOT="${PROJECT_OUTPUT_ROOT}/ranke_llamagen"
EXP_ROOT="${TASK_ROOT}/${RUN_NAME:-base_model_evaluation}"
SAMPLES_ROOT="${EXP_ROOT}/samples_hpsv2"

# Resolve the combo mode
case "$COMBO_ID" in
    0) GPT_T="base";             VQ_T="base";            GPT_TAG="gpt-base"; VQ_TAG="vq-base" ;;
    1) GPT_T="gpt_finetuned.pt"; VQ_T="vq_finetuned.pt"; GPT_TAG="gpt-onl";  VQ_TAG="vq-onl"  ;;
    2) GPT_T="gpt_finetuned.pt"; VQ_T="vq_ema.pt";       GPT_TAG="gpt-onl";  VQ_TAG="vq-ema"  ;;
    3) GPT_T="gpt_ema.pt";       VQ_T="vq_finetuned.pt"; GPT_TAG="gpt-ema";  VQ_TAG="vq-onl"  ;;
    4) GPT_T="gpt_ema.pt";       VQ_T="vq_ema.pt";       GPT_TAG="gpt-ema";  VQ_TAG="vq-ema"  ;;
    5) GPT_T="gpt_finetuned.pt"; VQ_T="base";            GPT_TAG="gpt-onl";  VQ_TAG="vq-base" ;;
esac

echo "========================================================================"
echo ">>> HPSv2 Sampling Started"
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
# [4] Main loop: sampling only
# ==============================================================================
for CURRENT_CFG in "${CFG_LIST[@]}"; do
    for step in "${loop_steps[@]}"; do
        echo ">>> ---------------------------------------------------"
        
        # Path-resolution and fallback logic
        if [ "$step" == "base" ]; then
            GPT_PATH="$GPT_CKPT_PATH_STAGE1"
            FINAL_VQ_PATH="$VQ_CKPT_PATH"
            ACTUAL_VQ_TAG="vq-base"
            OUT_DIR_NAME="baseline_gpt-base_vq-base_cfg${CURRENT_CFG}"
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
            OUT_DIR_NAME="sample_step${step}_${GPT_TAG}_${ACTUAL_VQ_TAG}_cfg${CURRENT_CFG}"
        fi

        [ ! -f "$GPT_PATH" ] && echo ">>> [Error] GPT not found: $GPT_PATH" && continue

        SAMPLE_OUT_DIR="${SAMPLES_ROOT}/${OUT_DIR_NAME}"
        SCORE_FILE="${SAMPLE_OUT_DIR}/score.txt"
        
        # Launch sampling
        if [ -f "$SCORE_FILE" ]; then
            echo ">>> [Skip] Already completely evaluated: $OUT_DIR_NAME"
        else
            # Check whether sampling is already complete (the HPSv2 all-category set is about 3200 images)
            if [ ! -d "$SAMPLE_OUT_DIR" ] || [ $(find "$SAMPLE_OUT_DIR" -type f -name "*.jpg" | wc -l) -lt 3000 ]; then
                echo ">>> Running HPSv2 Sampling for Step: $step | CFG: $CURRENT_CFG"
                mkdir -p "$SAMPLE_OUT_DIR"
                
                # Use the dynamic $GPUS variable
                torchrun --nnodes=1 --nproc_per_node=$GPUS \
                    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
                    "$SAMPLING_SCRIPT" \
                    --gpt-ckpt "$GPT_PATH" \
                    --vq-ckpt "$FINAL_VQ_PATH" \
                    --t5-path "$T5_PATH" \
                    --save-dir "$SAMPLE_OUT_DIR" \
                    --hpdv2-path "$HPDV2_PATH" \
                    --gpt-model "GPT-XL" \
                    --image-size "$IMAGE_SIZE" \
                    --cfg-scale "$CURRENT_CFG" \
                    --batch-size "$BATCH_SIZE" \
                    --precision "bf16"
            else
                echo ">>> [Notice] Sampling seems complete for $OUT_DIR_NAME, waiting for evaluation."
            fi
        fi

        [ "$step" == "base" ] && break
    done
done

echo ">>> All HPSv2 Sampling Sweeps Completed."
