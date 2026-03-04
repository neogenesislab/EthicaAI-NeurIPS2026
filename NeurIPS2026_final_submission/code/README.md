# EthicaAI — Code & Reproduction

> From Situational to Unconditional Commitment in Multi-Agent Social Dilemmas with Tipping Points

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick smoke test (~12 seconds)
python reproduce_quick.py --fast

# Full reproduction (~15 minutes, CPU)
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
├── reproduce_quick.py           # Entry point for reproduction
├── robustness_experiments.py    # CVaR, partial obs, adaptive adversaries
├── requirements.txt             # Python dependencies (pinned versions)
├── Dockerfile                   # Reproducible environment
├── LICENSE                      # MIT License
├── scripts/
│   ├── envs/
│   │   └── nonlinear_pgg_env.py       # Gymnasium-style PGG environment
│   ├── ppo_nash_trap.py               # Ind. REINFORCE (Linear/MLP/Critic)
│   ├── cleanrl_mappo_pgg.py           # CleanRL IPPO/MAPPO baselines (20 seeds)
│   ├── cleanrl_qmix_pgg.py            # CleanRL QMIX baseline (20 seeds)
│   ├── hp_sweep_ippo.py               # HP sensitivity (20 combos × 10 seeds)
│   ├── phi1_ablation.py               # φ₁ sweep (20 seeds)
│   ├── scale_test_n100.py             # N=100 scale test
│   ├── dnn_ablation.py                # Network depth ablation
│   ├── kpg_experiment.py              # K-level anticipation
│   ├── meta_learn_g.py                # Meta-learning g(θ,R)
│   ├── spatial_dilemma.py             # Spatial social dilemma
│   ├── cpr_experiment.py              # CPR cross-environment
│   ├── partial_obs_experiment.py      # Partial observability
│   ├── inject_tables.py               # JSON → LaTeX table injection
│   ├── audit_submission.py            # Submission integrity checker (8 modules)
│   └── build_submission_zip.py        # ZIP packager
└── outputs/                     # Experiment results (JSON)
    ├── cleanrl_baselines/
    │   ├── cleanrl_baseline_results.json   # IPPO/MAPPO results
    │   ├── qmix_baseline_results.json      # QMIX results
    │   └── hp_sweep_results.json           # HP sweep results
    ├── phi1_ablation/phi1_results.json
    ├── ppo_nash_trap/
    ├── scale_n100/
    ├── dnn_ablation/
    ├── kpg_experiment/
    ├── meta_learn_g/
    ├── partial_obs/
    └── round2/
```

## Experiments

All experiments use N=20 agents, E=20.0 endowment, 30% Byzantine adversaries, and 20 seeds unless noted.

| Experiment | Script | Seeds | Output |
|---|---|---|---|
| Ind. REINFORCE (Table 3) | `ppo_nash_trap.py` | 5 | `outputs/ppo_nash_trap/` |
| IPPO/MAPPO (Table 3) | `cleanrl_mappo_pgg.py` | 20 | `outputs/cleanrl_baselines/cleanrl_baseline_results.json` |
| QMIX (Table 3) | `cleanrl_qmix_pgg.py` | 20 | `outputs/cleanrl_baselines/qmix_baseline_results.json` |
| HP Sweep (Appendix) | `hp_sweep_ippo.py` | 10×20 | `outputs/cleanrl_baselines/hp_sweep_results.json` |
| φ₁ Ablation (Table 5) | `phi1_ablation.py` | 20 | `outputs/phi1_ablation/phi1_results.json` |
| Scale N=100 (Table 4) | `scale_test_n100.py` | 10 | `outputs/scale_n100/` |
| DNN Ablation | `dnn_ablation.py` | 5 | `outputs/dnn_ablation/` |
| KPG | `kpg_experiment.py` | 5 | `outputs/kpg_experiment/` |

## Verification

```bash
# Run submission integrity audit (8 modules)
python scripts/audit_submission.py

# Inject latest JSON data into LaTeX tables
python scripts/inject_tables.py
```

## Requirements

- Python ≥ 3.8
- NumPy, Matplotlib (see `requirements.txt`)
- No GPU required; all experiments run on a single CPU core

## License

MIT License. See [LICENSE](LICENSE) for details.
