#!/usr/bin/env bash
# Sequential ablation runner for the ELF-PT comparison study.
#
# Runs 7 configurations end-to-end:
#   01_elf_baseline       — vanilla ELF (K=1, no thoughts)
#   02_pt_r_k1_div        — ELF-PT-R K_reasoning=1 with diversity loss
#   03_pt_r_k2_div        — ELF-PT-R K_reasoning=2 with diversity loss
#   04_pt_r_k4_div        — ELF-PT-R K_reasoning=4 with diversity loss
#   05_pt_r_k1_nodiv      — ELF-PT-R K_reasoning=1 without diversity loss
#   06_pt_r_k2_nodiv      — ELF-PT-R K_reasoning=2 without diversity loss
#   07_pt_r_k4_nodiv      — ELF-PT-R K_reasoning=4 without diversity loss
#
# Each run is time-limited via the TIMEOUT_PER_RUN env var (default 5400 = 90 min).
# Total wall clock: ~7 * (TIMEOUT_PER_RUN / 3600) hours.
#
# Usage:
#   WANDB_API_KEY=<key> bash scripts/run_ablation.sh
#
# Designed to be launched inside tmux for unattended overnight runs.

set -euo pipefail

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY must be set in the environment." >&2
    echo "       Launch with: WANDB_API_KEY=<key> bash scripts/run_ablation.sh" >&2
    exit 1
fi

REPO="${REPO:-/localhome/local-chrislin/ELF-PT}"
HF_CACHE="${HF_CACHE:-/localhome/local-chrislin/.cache/huggingface}"
TIMEOUT_PER_RUN="${TIMEOUT_PER_RUN:-5400}"

CONFIGS=(
    "01_elf_baseline"
    "02_pt_r_k1_div"
    "03_pt_r_k2_div"
    "04_pt_r_k4_div"
    "05_pt_r_k1_nodiv"
    "06_pt_r_k2_nodiv"
    "07_pt_r_k4_nodiv"
)

mkdir -p "$REPO/outputs/ablation"

# Master log records the wall-clock summary across runs.
MASTER_LOG="$REPO/outputs/ablation/master.log"
echo "=== ABLATION RUN STARTED $(date) ===" | tee -a "$MASTER_LOG"
echo "TIMEOUT_PER_RUN=${TIMEOUT_PER_RUN}s" | tee -a "$MASTER_LOG"
echo "Configs: ${CONFIGS[*]}" | tee -a "$MASTER_LOG"

for cfg in "${CONFIGS[@]}"; do
    echo "" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"
    echo "[$(date)] Starting $cfg" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"

    RUN_LOG="$REPO/outputs/ablation/${cfg}.log"
    CONTAINER_NAME="elf_pt_ablation_${cfg}"

    # If a previous container with the same name lingers, remove it first.
    sudo docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    # Run the training, capturing stdout+stderr to both the per-run log and master tee.
    sudo docker run --rm --gpus all \
        --name "$CONTAINER_NAME" \
        -v "$REPO":/workspace \
        -v "$HF_CACHE":/cache/hf \
        -e HF_HOME=/cache/hf \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        elf-pt:smoke \
        bash -c "cd /workspace/src && timeout ${TIMEOUT_PER_RUN} python train.py --config configs/training_configs/ablation/${cfg}.yml" \
        > "$RUN_LOG" 2>&1 || true

    EXIT=$?
    echo "[$(date)] $cfg exited with code ${EXIT}" | tee -a "$MASTER_LOG"

    # Brief tail of the run log for sanity
    echo "--- last 20 lines of $cfg log ---" | tee -a "$MASTER_LOG"
    tail -20 "$RUN_LOG" | tee -a "$MASTER_LOG"
    echo "--- end tail ---" | tee -a "$MASTER_LOG"

    # Small gap so the GPU memory frees and the next run starts clean.
    sleep 10
done

echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "[$(date)] ALL ABLATION RUNS COMPLETE" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
