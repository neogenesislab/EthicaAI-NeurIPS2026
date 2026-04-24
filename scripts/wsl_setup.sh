#!/bin/bash
# =============================================================
# EthicaAI WSL2 JAX GPU 환경 자동 셋업 스크립트
# RTX 4070 SUPER (12GB) + CUDA 12 + JAX GPU
# =============================================================
set -e

echo "=========================================="
echo "EthicaAI WSL2 GPU Setup"
echo "=========================================="

# 1. 시스템 패키지 업데이트
echo "[1/5] System update..."
sudo apt-get update -qq && sudo apt-get upgrade -y -qq

# 2. Python 3.12 + pip 설치
echo "[2/5] Installing Python 3.12..."
sudo apt-get install -y -qq python3.12 python3.12-venv python3-pip

# 3. 가상환경 생성
echo "[3/5] Creating virtual environment..."
VENV_DIR="$HOME/ethica_env"
python3.12 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# 4. JAX CUDA 12 설치 (GPU 지원)
echo "[4/5] Installing JAX with CUDA 12..."
pip install --upgrade pip
pip install "jax[cuda12]" scipy matplotlib

# 5. GPU 검증
echo "[5/5] Verifying GPU access..."
python3 -c "
import jax
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
print('Backend:', jax.default_backend())
print('GPU available:', any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in jax.devices()))
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "To run experiments:"
echo "  source ~/ethica_env/bin/activate"
echo "  cd /mnt/d/00.test/PAPER/EthicaAI_anon2"
echo "  python simulation/jax/run_full_pipeline.py baseline"
echo "=========================================="
