---
description: NeurIPS 2026 보강 실험 실행 (100-에이전트 Full Sweep)
---

# NeurIPS 2026 보강 실험 실행 워크플로우

> E1 실험 보강을 위한 실행 순서

## 사전 준비

WSL2 환경에서 ethica_env 활성화 필요.

## 실행 순서

// turbo-all

### 1. 100-에이전트 Full Model (7 SVO × 10 seeds = 70 runs)
```bash
wsl --distribution Ubuntu-24.04 -- bash -c "source ~/ethica_env/bin/activate && cd /mnt/d/00.test/PAPER/EthicaAI_anon2 && python -m simulation.jax.run_full_pipeline large_full"
```
예상 소요: ~70 × 30초 ≈ 35분 (GPU)

### 2. 100-에이전트 Baseline (Meta-Ranking OFF)
```bash
wsl --distribution Ubuntu-24.04 -- bash -c "source ~/ethica_env/bin/activate && cd /mnt/d/00.test/PAPER/EthicaAI_anon2 && python -m simulation.jax.run_full_pipeline large_baseline"
```

### 3. 100-에이전트 Harvest 환경
```bash
wsl --distribution Ubuntu-24.04 -- bash -c "source ~/ethica_env/bin/activate && cd /mnt/d/00.test/PAPER/EthicaAI_anon2 && python -m simulation.jax.run_full_pipeline large_harvest"
```

### 4. 결과 재분석 (LMM + Bootstrap CI)
```bash
wsl --distribution Ubuntu-24.04 -- bash -c "source ~/ethica_env/bin/activate && cd /mnt/d/00.test/PAPER/EthicaAI_anon2 && python -m simulation.jax.reanalyze <run_dir>"
```

## 결과 확인

실험 결과는 `simulation/outputs/run_large_<timestamp>/` 에 저장됨:
- `sweep_large_<timestamp>.json` — 원시 데이터
- `causal_results.json` — 인과분석 결과
- `figures/` — 생성된 Figure들
- `summary.json` — 전체 요약
