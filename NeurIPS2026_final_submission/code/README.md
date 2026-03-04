# EthicaAI: The Moral Commitment Spectrum

> From Situational to Unconditional Commitment in Multi-Agent Social Dilemmas with Tipping Points

## Overview

This repository contains the code and data for reproducing all experiments in the paper. We study how agents should adjust their prosocial behavior (commitment level λ) in social dilemmas with non-linear dynamics and tipping points.

**Key Finding**: In environments with tipping points, standard MARL algorithms (REINFORCE, IPPO, MAPPO, QMIX) converge to suboptimal "Nash Trap" equilibria. Only *unconditional commitment* (φ₁* = 1.0) prevents collapse.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick reproduction (~2 min)
python reproduce_quick.py --fast

# Full reproduction (~30 min)
python reproduce_quick.py
```

## Docker

```bash
docker build -t ethicaai .
docker run ethicaai
```

## Project Structure

```
code/
├── reproduce_quick.py       # Entry point for reproduction
├── requirements.txt         # Python dependencies
├── Dockerfile               # Reproducible environment
├── LICENSE                  # MIT License
└── scripts/
    ├── envs/
    │   └── nonlinear_pgg_env.py    # Gymnasium PGG environment
    ├── cleanrl_mappo_pgg.py        # CleanRL IPPO/MAPPO baselines (20 seeds)
    ├── cleanrl_qmix_pgg.py         # CleanRL QMIX baseline (20 seeds)
    ├── hp_sweep_ippo.py            # HP sensitivity sweep (12 combos × 3 seeds)
    ├── round2_experiments.py       # Ablation & extended experiments
    ├── compute_radar_scores.py     # 5-axis radar chart computation
    ├── generate_tables.py          # JSON → LaTeX table generation
    └── verify_submission.py        # Submission integrity checker

outputs/
├── cleanrl_baselines/
│   ├── cleanrl_baseline_results.json   # IPPO/MAPPO: 20 seeds each
│   ├── qmix_baseline_results.json      # QMIX: 20 seeds
│   └── hp_sweep_results.json           # HP sweep: 12×3=36 runs
├── round2/
│   └── round2_results.json             # Ablation results
└── radar_scores.json                   # 5-axis comparison scores
```

## Experiments

All experiments use N=20 agents, E=20.0 endowment, 30% Byzantine adversaries, and 20 seeds unless noted.

| Experiment | Script | Seeds | Output |
|---|---|---|---|
| IPPO/MAPPO baselines | `cleanrl_mappo_pgg.py` | 20 | `cleanrl_baseline_results.json` |
| QMIX baseline | `cleanrl_qmix_pgg.py` | 20 | `qmix_baseline_results.json` |
| HP sensitivity | `hp_sweep_ippo.py` | 3×12 | `hp_sweep_results.json` |
| Ablation | `round2_experiments.py` | varies | `round2_results.json` |

## Verification

```bash
# Check submission integrity
python scripts/verify_submission.py

# Generate LaTeX tables from JSON
python scripts/generate_tables.py

# Compute radar scores
python scripts/compute_radar_scores.py
```

## Requirements

- Python ≥ 3.8
- NumPy, Gymnasium, Torch (see `requirements.txt`)
- No GPU required; all experiments run on a single CPU core

## License

MIT License. See [LICENSE](LICENSE) for details.
