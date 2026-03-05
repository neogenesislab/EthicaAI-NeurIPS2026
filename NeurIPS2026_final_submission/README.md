# EthicaAI: The Moral Commitment Spectrum

> **From Situational to Unconditional: The Spectrum of Moral Commitment Required for Multi-Agent Survival in Non-linear Social Dilemmas**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](code/LICENSE)

---

## Abstract

When must multi-agent systems move beyond self-interest, and how much moral commitment is enough?
We investigate this question through a systematic empirical study of cooperation dynamics in Public Goods Games (PGG) of increasing environmental severity, drawing interpretive framing from Amartya Sen's meta-ranking theory.

### Key Contributions

1. **(C1) Computational Meta-Ranking**: MARL formalization of Sen's theory with dynamic commitment λₜ ∈ [0,1] conditioned on resource state and SVO
2. **(C2) Situational Commitment in Linear Environments**: Group-level ESS (x̄=0.987) outperforming 8 baselines including M-FOS and POLA
3. **(C3) Algorithm-Invariant Cooperation Failure**: Independent RL agents (Linear, MLP, Actor-Critic, IPPO, MAPPO, QMIX) all converge to suboptimal equilibria — though QMIX achieves notably higher cooperation (71.8%) than IPPO/MAPPO (~37%), it still falls short of the oracle (100%)
4. **(C4) Unconditional Commitment + Meta-Learning Validation**: Only φ₁*=1.0 prevents collapse. Meta-learning independently recovers this optimum.

---

## Repository Structure

```
NeurIPS2026_final_submission/
├── paper/                          # LaTeX source + compiled PDF
│   ├── unified_paper.tex
│   ├── unified_paper.pdf
│   ├── unified_references.bib
│   └── *.png                       # Figures
├── code/
│   ├── reproduce_quick.py          # Entry point for reproduction
│   ├── requirements.txt            # Python dependencies (pinned)
│   ├── Dockerfile                  # Reproducible environment
│   ├── LICENSE                     # MIT License
│   ├── robustness_experiments.py   # CVaR, partial obs, adaptive adversaries
│   ├── scripts/
│   │   ├── envs/
│   │   │   └── nonlinear_pgg_env.py    # Gymnasium-style PGG environment
│   │   ├── ppo_nash_trap.py            # Ind. REINFORCE (Linear/MLP/Critic)
│   │   ├── cleanrl_mappo_pgg.py        # CleanRL IPPO/MAPPO (20 seeds)
│   │   ├── cleanrl_qmix_pgg.py         # CleanRL IQL (20 seeds)
│   │   ├── hp_sweep_ippo.py            # HP sensitivity (20 combos × 10 seeds)
│   │   ├── phi1_ablation.py            # φ₁ sweep (20 seeds)
│   │   ├── scale_test_n100.py          # N=100 scale test
│   │   ├── dnn_ablation.py             # Network depth ablation
│   │   ├── kpg_experiment.py           # K-level anticipation
│   │   ├── meta_learn_g.py             # Meta-learning g(θ,R)
│   │   ├── spatial_dilemma.py          # Spatial social dilemma
│   │   ├── cpr_experiment.py           # CPR cross-environment
│   │   ├── partial_obs_experiment.py   # Partial observability
│   │   ├── inject_tables.py            # JSON → LaTeX table injection
│   │   ├── audit_submission.py         # Submission integrity checker
│   │   └── build_submission_zip.py     # ZIP packager
│   └── outputs/                    # Experiment results (JSON)
│       ├── cleanrl_baselines/
│       ├── phi1_ablation/
│       ├── ppo_nash_trap/
│       ├── scale_n100/
│       ├── dnn_ablation/
│       ├── kpg_experiment/
│       ├── meta_learn_g/
│       ├── partial_obs/
│       └── round2/
└── README.md                       # This file
```

---

## Quick Start

```bash
# Install dependencies
cd code
pip install -r requirements.txt

# Quick smoke test (~12 seconds)
python reproduce_quick.py --fast

# Full reproduction (~15 minutes, CPU)
python reproduce_quick.py
```

---

## Reproducing Key Results

| Table/Figure | Script | Output |
|:---|:---|:---|
| Table 3 (Nash Trap) | `ppo_nash_trap.py`, `cleanrl_mappo_pgg.py`, `cleanrl_qmix_pgg.py` | `outputs/ppo_nash_trap/`, `outputs/cleanrl_baselines/` |
| Table 4 (Scale N=100) | `scale_test_n100.py` | `outputs/scale_n100/` |
| Table 5 (φ₁ Sweep) | `phi1_ablation.py` | `outputs/phi1_ablation/phi1_results.json` |
| Table 6 (Meta-Learn) | `meta_learn_g.py` | `outputs/meta_learn_g/` |
| HP Sweep (Appendix) | `hp_sweep_ippo.py` | `outputs/cleanrl_baselines/hp_sweep_results.json` |
| DNN Ablation | `dnn_ablation.py` | `outputs/dnn_ablation/` |
| KPG Experiment | `kpg_experiment.py` | `outputs/kpg_experiment/` |
| Robustness (CVaR, PO, Adv) | `robustness_experiments.py` | `outputs/round2/` |

---

## License

MIT License — see `code/LICENSE` for details.
