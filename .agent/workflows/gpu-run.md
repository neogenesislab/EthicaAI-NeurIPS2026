---
description: WSL2 GPU로 EthicaAI 시뮬레이션 실행
---
// turbo-all

## WSL2 GPU 실행

1. 전체 reproduce 파이프라인 (GPU)
```bash
wsl -d Ubuntu-24.04 -- bash -c "source ~/ethicaai_env/bin/activate && cd /mnt/d/00.test/PAPER/EthicaAI_anon2 && python3 reproduce.py 2>&1"
```

2. 특정 Phase만 실행
```bash
wsl -d Ubuntu-24.04 -- bash -c "source ~/ethicaai_env/bin/activate && cd /mnt/d/00.test/PAPER/EthicaAI_anon2 && python3 reproduce.py --phase P 2>&1"
```

3. 단일 모듈 실행 (예: scale_1000)
```bash
wsl -d Ubuntu-24.04 -- bash -c "source ~/ethicaai_env/bin/activate && cd /mnt/d/00.test/PAPER/EthicaAI_anon2 && python3 -m simulation.jax.analysis.scale_1000 simulation/outputs/reproduce 2>&1"
```

4. GPU 상태 확인
```bash
wsl -d Ubuntu-24.04 -- bash -c "nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader"
```

5. JAX GPU 벤치마크
```bash
wsl -d Ubuntu-24.04 -- bash /mnt/d/00.test/PAPER/EthicaAI_anon2/scripts/test_gpu.sh
```

6. Genesis v2.0 진화 루프 (GPU)
```bash
wsl -d Ubuntu-24.04 -- bash /mnt/d/00.test/PAPER/EthicaAI_anon2/scripts/run_evolution_gpu.sh
```
