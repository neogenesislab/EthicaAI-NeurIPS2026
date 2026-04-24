#!/bin/bash
# EthicaAI Full Sweep Runner — WSL2 GPU 백그라운드 실행용
# 사용법: wsl -d Ubuntu-24.04 -- bash /mnt/d/00.test/PAPER/EthicaAI_anon2/scripts/run_full_sweep.sh
set -e

echo "=== EthicaAI Full Sweep Starting ==="
echo "Time: $(date)"

# 가상환경 활성화
source ~/ethicaai_env/bin/activate

# 프로젝트 디렉토리 이동
cd /mnt/d/00.test/PAPER/EthicaAI_anon2

# 로그 디렉토리 생성
mkdir -p experiments/full_sweep_results

echo "Python: $(which python3)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# Full Sweep 실행
python3 scripts/full_sweep_phase1.py

echo "=== Full Sweep Completed ==="
echo "Time: $(date)"
