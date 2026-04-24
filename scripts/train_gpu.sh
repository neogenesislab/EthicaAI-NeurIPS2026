#!/bin/bash
# EthicaAI 100-에이전트 Full Sweep 학습
# RTX 4070 SUPER (12GB) — WSL2 Ubuntu 24.04
# 7 SVO × 10 seeds × 2 (Meta ON/OFF) = 140 runs
set -e

source ~/ethicaai_env/bin/activate
cd /mnt/d/00.test/PAPER/EthicaAI_anon2

export PYTHONIOENCODING=utf-8
export PYTHONPATH=/mnt/d/00.test/PAPER/EthicaAI_anon2

DEVICE=$(python3 -c "import jax; print(jax.devices()[0])")
echo "============================================================"
echo "  EthicaAI NeurIPS 100-Agent Full Sweep"
echo "  Device: $DEVICE"
echo "  Plan: large_full(70) + large_baseline(70) = 140 runs"
echo "============================================================"

START=$(date +%s)

# 1. Meta-Ranking ON: 7 SVO × 10 seeds
echo ""
echo "[1/2] large_full (Meta ON, 70 runs) ..."
python3 -m simulation.jax.run_full_pipeline large_full 2>&1

MID=$(date +%s)
echo "  [1/2] Done in $((MID-START))s"

# 2. Baseline: Meta-Ranking OFF
echo ""
echo "[2/2] large_baseline (Meta OFF, 70 runs) ..."
python3 -m simulation.jax.run_full_pipeline large_baseline 2>&1

END=$(date +%s)
echo "  [2/2] Done in $((END-MID))s"

echo ""
echo "============================================================"
echo "  FULL SWEEP COMPLETE"
echo "  Total: $((END-START))s"
echo "============================================================"
