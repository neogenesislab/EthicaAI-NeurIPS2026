# 🧬 EthicaAI: Beyond Homo Economicus

> **Computational Verification of Amartya Sen's Meta-Ranking Theory in Multi-Agent Social Dilemmas**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18728438-blue?style=for-the-badge&logo=zenodo)](https://doi.org/10.5281/zenodo.18728438)
[![NeurIPS 2026](https://img.shields.io/badge/Target-NeurIPS_2026-purple?style=for-the-badge)](https://neurips.cc)
[![Engine](https://img.shields.io/badge/Engine-JAX_GPU-red?style=for-the-badge&logo=nvidia)](https://jax.readthedocs.io)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://python.org)

---

## 📄 Abstract

Can artificial agents develop genuine moral commitment beyond self-interest? We formalize Amartya Sen's **Meta-Ranking** theory — preferences over preferences implementing moral commitment — within a Multi-Agent Reinforcement Learning (MARL) framework. Our dynamic commitment mechanism λₜ, conditioned on resource availability and Social Value Orientation (SVO), is tested across **8 environments** with up to **1,000 agents**.

### Key Contributions

1. **Dynamic Meta-Ranking (C1)**: First MARL implementation of Sen's theory with convergence guarantees
2. **Role Specialization (C2)**: Discovery of emergent division of labor resolving the "cooperation paradox"
3. **Local Optimality (C3)**: Situational Commitment matches Utilitarian welfare using only local information
4. **Scale & Robustness (C4)**: Verified across 1,000 agents, 5 topologies, and 50% adversarial populations
5. **SOTA Baseline — LOLA (C5)**: JAX-based Exact LOLA comparison revealing O(N³) collapse and 0.209 cooperation ceiling
6. **SOTA Baseline — IPPO (C6)**: Independent PPO shows learning paralysis (≈0.5), proving deep RL alone cannot discover cooperation

---

## 🔬 Benchmark Results

This repository contains comprehensive SOTA baseline comparisons demonstrating the fundamental limitations of existing approaches in N-player settings.

### Meta-Ranking vs. Exact LOLA (N=20 PGG)

| Dimension | Meta-Ranking | Exact LOLA | Ratio |
|:---|:---:|:---:|:---:|
| **Cooperation** | **0.576** ± 0.039 | 0.209 (best of 25) | 2.75× |
| **Compute (per step)** | 34ms | 12,600ms | **370×** faster |
| **Byzantine (50%)** | **0.209** | 0.124 | 1.69× |
| **N=100 Scaling** | ✅ (34ms) | ❌ OOM crash | — |

### Meta-Ranking vs. Independent PPO (N=20 PGG, 5 seeds)

| Method | Coop Rate | Time | Ratio |
|:---|:---:|:---:|:---:|
| **Meta-Ranking** | **0.576** ± 0.039 | 10s | — |
| Prosocial IPPO | 0.520 ± 0.019 | 7,049s | **680×** slower |
| Selfish IPPO | 0.518 ± 0.019 | 7,073s | **680×** slower |

Both IPPO variants exhibit **learning paralysis** (≈sigmoid(0)=0.5), failing to escape initialization despite gradient-based learning.

### Benchmark Data

- [`grid_search_results.json`](submission_package/paper/grid_search_results.json) — 25-config LOLA grid search
- [`neurips2026_main.pdf`](submission_package/paper/neurips2026_main.pdf) — Compiled 7-page manuscript

---

## 💻 Quick Start

```bash
# Clone
git clone https://github.com/Yesol-Pilot/EthicaAI.git
cd EthicaAI

# Setup (requires NVIDIA GPU + WSL2)
bash scripts/setup_env.sh

# Run simulations
bash scripts/run_evolution_gpu.sh
```

---

## 📂 Repository Structure

```
EthicaAI/
├── submission_package/  # 📦 NeurIPS 2026 submission (main)
│   └── paper/           #   LaTeX source + compiled PDF
├── paper/               # 📄 Extended manuscript
├── simulation/          # 🎮 JAX simulation core
├── experiments/         # 🧪 Experiment configs & results
├── scripts/             # 🛠️ Automation scripts
└── analysis/            # 📊 Analysis notebooks
```

---

## 📝 Citation

```bibtex
@software{ethicaai2026,
  title   = {Beyond Homo Economicus: Computational Verification of
             Amartya Sen's Meta-Ranking Theory in Multi-Agent Social Dilemmas},
  author  = {Yesol Heo},
  year    = {2026},
  doi     = {10.5281/zenodo.18728438},
  url     = {https://github.com/Yesol-Pilot/EthicaAI},
  version = {v1.1.0-preprint}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

**Author**: [Yesol Heo](https://heoyesol.kr)
