#!/bin/bash
#SBATCH --job-name=ranke_llamagen_train_hps
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512GB

set -e
set -x

# ==============================================================================
# [1] Core configuration (edit here first)
# ==============================================================================

# --- A. Weight source and experiment naming ---
# Initial model source
SFT_SOURCE_RUN=""              # Leave empty to start from the base model; set a public SFT run name to start from that checkpoint
SFT_SOURCE_STEP="0"         # SFT checkpoint step

# Example: RESUME_DIR="/path/to/outputs/ranke_llamagen/run_name_example/checkpoint_500"
# Leave this empty ("") when starting a brand-new RL run
RESUME_DIR=""

TRAIN_MODE="both"               # Training mode: both, gpt, or decoder
EXP_PREFIX="ranke_${TRAIN_MODE}_hpdv2"       # Experiment name prefix (HPDv2 data)
SCALING_SIZE="15k"             # Dataset scale (HPDv2 15k subset)
WANDB_PROJ="ranke_llamagen"

# --- B. Optimizer hyperparameters ---
LR_GPT=1e-5                    # GPT learning rate
LR_DECODER=5e-5                # Decoder learning rate
LR_DISC=2e-5                   # Discriminator learning rate
MAX_EPOCHS=5                  # Total training epochs
BATCH_SIZE_PER_GPU=4           # Batch size per GPU
GRAD_ACCUM=1                   # Gradient accumulation steps

# --- C. Hybrid training weights (RL and decoder losses) ---
KL_COEF=0.05                   # GRPO KL penalty
LAMBDA_DEC_GAN=1.0             # GAN loss weight
LAMBDA_DEC_REWARD=0.1         # Direct reward-backprop weight                   ******** key setting ********
LAMBDA_RECON=1.0               # Ground-truth reconstruction weight
LAMBDA_DEC_CONSISTENCY=10.0     # Decoder consistency loss weight (static when schedule_type=none)

# --- Decoder consistency scheduling ---
CONSISTENCY_SCHEDULE_TYPE="none"  # Schedule type: none / linear / sin
CONSISTENCY_START_STEP=0            # Schedule start step
CONSISTENCY_END_STEP=6000           # Schedule end step
CONSISTENCY_START_VALUE=0.0         # Initial lambda_decoder_consistency value
CONSISTENCY_END_VALUE=20.0          # Final lambda_decoder_consistency value

# Consistency schedule
# 20.0 |                  ___________
#     |               /
#     |            /
#     |         /
#     |      /
# 0.0 |___/
#     +-----|----------|----------->
#           0       6000        step


# --- Four reward weights (adjust as needed) ---
R_CLIP=0.0
R_AESTHETIC=0.0
R_IMAGE_REWARD=0.0
R_HPSV2=1.0

# --- D. Sampling and annealing ---
REJECTION_K=1                  # Rejection-sampling K
LASC_K=20                      # Number of LASC samples
TEMP_START=4.0                 # Sampling annealing start temperature
TEMP_END=1.0                   # Sampling annealing end temperature
ANNEAL_RATIO=0.2               # Annealing ratio (0.2 means annealing completes in the first 20% of steps)

# --- E. Generation and EMA configuration ---
GEN_CFG=6.0                    # CFG scale used during generation
GEN_TEMP=1.0                   # Generation temperature
DISC_TYPE="patchgan"           # Discriminator type
EMA_DECAY_GPT=0.999            # GPT EMA decay
EMA_DECAY_VQ=0.99              # VQ EMA decay

# --- F. Logging and checkpoint frequency ---
LOG_INTERVAL=1                 # Logging interval
SAVE_INTERVAL=500              # Checkpoint save interval
SAMPLING_INTERVAL=500          # Validation sampling interval

# ==============================================================================
# [2] Environment setup and path logic (usually no edits needed)
# ==============================================================================

# 2.1 Load the shared environment config
# 2.1 Load the shared environment config
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

# 2.2 Resolve model paths automatically (resume mode uses these only as fallback/logging info)
if [ -z "$SFT_SOURCE_RUN" ]; then
    GPT_TARGET_PATH="$GPT_CKPT_PATH_STAGE1"
    SOURCE_DESC="Base Pre-trained Model"
    EXP_SUFFIX="from_base"
else
    SFT_BASE_ROOT="${PROJECT_OUTPUT_ROOT}/ranke_llamagen_sft"
    GPT_TARGET_PATH="${SFT_BASE_ROOT}/${SFT_SOURCE_RUN}/checkpoint_${SFT_SOURCE_STEP}/gpt_finetuned.pt"
    SOURCE_DESC="SFT Checkpoint (Run: ${SFT_SOURCE_RUN}, Step: ${SFT_SOURCE_STEP})"
    EXP_SUFFIX="from_sft"
    [ ! -f "$GPT_TARGET_PATH" ] && echo ">>> [Error] SFT Checkpoint NOT FOUND" && exit 1
fi

# 2.3 Build dataset paths (HPDv2)
SCALING_DATA_ROOT="${HPDV2_TRAIN_ROOT}"
TARGET_TAR_PATH="${SCALING_DATA_ROOT}/train_subset_${SCALING_SIZE}_hpdv2.tar"
[ ! -f "$TARGET_TAR_PATH" ] && echo "Error: HPDv2 Dataset tar not found: $TARGET_TAR_PATH" && exit 1

# 2.4 Distributed configuration
export OMP_NUM_THREADS=8
export ACCELERATE_MIXED_PRECISION="bf16"
export PYTHONPATH="${CODE_ROOT}/evaluations/HPSv2:${CODE_ROOT}:${PYTHONPATH}"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export GPUS_PER_NODE=8

# 2.5 Build experiment names and output directories
JOB_ID="${SLURM_JOB_ID:-local}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if [ -n "$RESUME_DIR" ]; then
    # When resuming, keep the original directory so logs and new checkpoints stay together
    OUTPUT_DIR=$(dirname "$RESUME_DIR")
    EXP_NAME=$(basename "$OUTPUT_DIR")
    echo ">>> [Notice] Resuming Training. Output dir will be: $OUTPUT_DIR"
else
    # For a new run, create a timestamped output directory
    EXP_NAME="${EXP_PREFIX}_${EXP_SUFFIX}_${TIMESTAMP}_${JOB_ID}"
    OUTPUT_DIR="${PROJECT_OUTPUT_ROOT}/${WANDB_PROJ}/${EXP_NAME}"
    mkdir -p "$OUTPUT_DIR/logs"
fi

# 2.6 Compute the global batch size
GLOBAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * GPUS_PER_NODE * SLURM_NNODES * GRAD_ACCUM))

# ==============================================================================
# [3] Assemble arguments and launch
# ==============================================================================

CMD_ARGS=(
    # Data & Paths
    --dataset-name "scaling_simple"
    --data-path "$TARGET_TAR_PATH"
    --output-dir "$OUTPUT_DIR"
    --gpt-ckpt "$GPT_TARGET_PATH"
    --vq-ckpt "$VQ_CKPT_PATH"
    --t5-path "$T5_PATH"
    
    # Training Loop Config
    --train-mode "$TRAIN_MODE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --batch-size "$BATCH_SIZE_PER_GPU"
    --gradient-accumulation-steps "$GRAD_ACCUM"
    --max-epochs "$MAX_EPOCHS"
    --mixed-precision "bf16"
    --sync-gpt-decoder-update
    --use-fixed-ref-model
    
    # Model Specs
    --image-size 256
    --gpt-model "GPT-XL"
    --vq-model "VQ-16"
    --t5-model-type "flan-t5-xl"
    
    # Optimizer
    --lr-gpt "$LR_GPT"
    --lr-decoder "$LR_DECODER"
    --lr-disc "$LR_DISC"
    
    # RL / GRPO
    --group-size 8
    --grpo-epochs 1
    --kl-coef "$KL_COEF"
    --rollout-length 256
    --grpo-adv-coef 1.0
    
    # Decoder Loss Weights
    --lambda-decoder-reward "$LAMBDA_DEC_REWARD"
    --lambda-decoder-gan "$LAMBDA_DEC_GAN"
    --lambda-reconstruction "$LAMBDA_RECON"
    --lambda-decoder-consistency "$LAMBDA_DEC_CONSISTENCY"
    
    # Decoder Consistency Scheduling
    --consistency-schedule-type "$CONSISTENCY_SCHEDULE_TYPE"
    --consistency-start-step "$CONSISTENCY_START_STEP"
    --consistency-end-step "$CONSISTENCY_END_STEP"
    --consistency-start-value "$CONSISTENCY_START_VALUE"
    --consistency-end-value "$CONSISTENCY_END_VALUE"
    
    # Sampling & Annealing
    --rejection-sample-k "$REJECTION_K"
    --lasc-sample-k "$LASC_K"
    --temp-start "$TEMP_START"
    --temp-end "$TEMP_END"
    --anneal-ratio "$ANNEAL_RATIO"
    
    # Discriminator & Rewards
    --disc-type "$DISC_TYPE"
    --dino-path "$DINO_PATH"
    --disc-start 100
    --disc-weight 0.5
    --reward-path-clip "$CLIP_PATH"
    --reward-path-aesthetic "$AES_REW_PATH"
    --reward-weight-clip "$R_CLIP"
    --reward-weight-aesthetic "$R_AESTHETIC"
    --reward-weight-image-reward "$R_IMAGE_REWARD"
    --reward-weight-hpsv2 "$R_HPSV2"
    
    # Gen & EMA
    --gen-cfg-scale "$GEN_CFG"
    --gen-temperature "$GEN_TEMP"
    --ema-decay-vq "$EMA_DECAY_VQ"
    --ema-decay-gpt "$EMA_DECAY_GPT"
    --frqs-ema-gpt-update 10
    
    # Logging & Saves
    --use-wandb
    --wandb-project "$WANDB_PROJ"
    --wandb-run-name "$EXP_NAME"
    --log-interval "$LOG_INTERVAL"
    --save-interval "$SAVE_INTERVAL"
    --sampling-steps "$SAMPLING_INTERVAL"
    --num-workers 8
)

# Inject resume arguments dynamically
if [ -n "$RESUME_DIR" ]; then
    if [ ! -d "$RESUME_DIR" ]; then
        echo ">>> [Error] RESUME_DIR not found: $RESUME_DIR"
        exit 1
    fi
    CMD_ARGS+=(--resume-from "$RESUME_DIR")
fi

echo "========================================================================"
echo ">>> Launching: $EXP_NAME"
echo ">>> Source: $SOURCE_DESC"
echo ">>> Data: $TARGET_TAR_PATH (HPDv2 ${SCALING_SIZE})"
if [ -n "$RESUME_DIR" ]; then
    echo ">>> Resume : $RESUME_DIR"
fi
echo "========================================================================"

torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    autoregressive/post_train/post_train.py \
    "${CMD_ARGS[@]}"