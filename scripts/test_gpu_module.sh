#!/bin/bash
# WSL2 GPU 단일 모듈 테스트
source ~/ethicaai_env/bin/activate
cd /mnt/d/00.test/PAPER/EthicaAI_anon2

echo "=== GPU Backend Check ==="
python3 -c "import jax; print(f'Backend: {jax.default_backend()}')"

echo ""
echo "=== P1: Scale 1000 ==="
python3 -m simulation.jax.analysis.scale_1000 simulation/outputs/reproduce 2>&1

echo ""
echo "=== DONE ==="
