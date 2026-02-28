# EthicaAI: When Should AI Agents Be Moral? 🧠⚖️

[![NeurIPS 2026](https://img.shields.io/badge/Target-NeurIPS_2026-blue?style=for-the-badge&logo=neurips)](https://neurips.cc)
[![74 Figures](https://img.shields.io/badge/Figures-74-brightgreen?style=for-the-badge)](https://ethicaai.vercel.app)
[![38 Modules](https://img.shields.io/badge/Modules-38-orange?style=for-the-badge)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-9cf?style=for-the-badge&logo=google&logoColor=white)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **"The question isn't *whether* AI should be moral, but *when*."**

**EthicaAI** formalizes Amartya Sen's **Meta-Ranking** theory (preferences over preferences) as a dynamic mechanism in Multi-Agent Reinforcement Learning. We demonstrate that _Situational Commitment_ — morality conditional on survival — is the only Evolutionarily Stable Strategy across 8 environments, 7 SVO conditions, and up to 1,000 agents.

<p align="center">
  <a href="https://ethicaai.vercel.app"><strong>🌐 Interactive Dashboard (74 Figures)</strong></a> &nbsp;|&nbsp;
  <a href="paper_english.md"><strong>📄 Full Paper (35 Sections)</strong></a> &nbsp;|&nbsp;
  <a href="paper/neurips2026_main.tex"><strong>📝 NeurIPS LaTeX</strong></a>
</p>

---

## 🔬 Key Findings

| # | Finding | Evidence |
|:-:|---------|---------|
| **1** | Dynamic meta-ranking (λ_t) significantly enhances collective welfare | p<0.001 (LMM), Cohen's f²=0.40 |
| **2** | Emergent **role specialization** (Cleaners vs Eaters) | σ divergence p<0.0001 |
| **3** | Only "Situational Commitment" survives as **ESS** | 5-theory comparison, 200-gen replicator |
| **4** | **Scale invariant** from 20 to 1,000 agents | SII ≈ 1.0, 1.32ms/agent |
| **5** | **Byzantine robust** at 50% adversarial population | Coop=1.000, 100% sustainability |
| **6** | SVO accounts for **79.8%** of λ_t determination | SHAP attribution analysis |

---

## 🛠️ Installation

```bash
# Clone & setup
git clone https://anonymous.4open.science/r/EthicaAI
cd EthicaAI

python -m venv ethica_env
source ethica_env/bin/activate  # Windows: ethica_env\Scripts\activate
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, CUDA 12+ (optional, for GPU acceleration)

---

## 🚀 Quick Start

### Reproduce All Results (One Command)
```bash
# All 38 analysis modules (Phase G → Q)
python reproduce.py

# Specific phase only
python reproduce.py --phase P  # Phase P (Scale, LMM, Mechanism Design, etc.)
python reproduce.py --phase Q  # Phase Q (Moran, GNN, Interpretability, Policy)

# Quick demo
python reproduce.py --quick
```

### Docker (Zero Setup)
```bash
# Build & run (generates all 74 figures)
docker build -t ethicaai .
docker run -v $(pwd)/output:/ethicaai/simulation/outputs/reproduce ethicaai

# Specific phase
docker run ethicaai python reproduce.py --phase P
```

### Run Individual Experiments
```bash
# Phase P: Deepening
python -m simulation.jax.analysis.scale_1000 simulation/outputs/reproduce       # P1: 1000-agent scale
python -m simulation.jax.analysis.lmm_causal_forest simulation/outputs/reproduce # P2: LMM + HTE
python -m simulation.jax.analysis.continuous_pgg simulation/outputs/reproduce     # P3: Continuous PGG
python -m simulation.jax.analysis.network_topology simulation/outputs/reproduce   # P4: Network effects
python -m simulation.jax.analysis.mechanism_design simulation/outputs/reproduce   # P5: IC/IR/NE
python -m simulation.jax.analysis.adversarial_robustness simulation/outputs/reproduce # P6: Byzantine

# Phase Q: Novel Contributions
python -m simulation.jax.analysis.moran_process simulation/outputs/reproduce     # Q2: Moran Process
python -m simulation.jax.analysis.moral_theories simulation/outputs/reproduce    # Q3: 5 Moral Theories
python -m simulation.jax.analysis.gnn_agent simulation/outputs/reproduce         # Q4: GNN Agents
python -m simulation.jax.analysis.interpretability simulation/outputs/reproduce  # Q5: Mechanistic
python -m simulation.jax.analysis.policy_implications simulation/outputs/reproduce # Q6: Policy
```

---

## 📊 74 Figures

All figures available at [ethicaai.vercel.app](https://ethicaai.vercel.app).

| Phase | Figures | Content |
|:-----:|:-------:|---------|
| **G** (Core) | 1–18 | Convergence, static/dynamic λ, sensitivity, cross-env |
| **H** (Evolution) | 9–14 | Evolutionary competition, mechanism decomposition |
| **M** (Extended) | 19–30 | Full sweep, mixed-SVO, communication, continuous PGG |
| **N** (Advanced) | 31–38 | MAPPO, partial obs, multi-resource, LLM comparison |
| **O** (Real-world) | 39–48 | Climate, vaccine, AI governance, Human-AI pilot |
| **P** (Deepening) | 49–62 | Scale 1000, LMM, continuous, network, mechanism, adversarial |
| **Q** (Novel) | 53–70 | Moral theories, Moran, interpretability, policy, GNN |

---

## 📂 Repository Structure

```
EthicaAI/
├── simulation/
│   └── jax/
│       ├── analysis/              # 38 analysis modules
│       ├── environments/          # Cleanup, IPD, PGG, Harvest, Network
│       ├── training/              # MAPPO training pipeline
│       └── run_full_pipeline.py   # End-to-end execution
├── paper/                         # NeurIPS 2026 LaTeX (Main 8p + Supplementary 30p)
├── site/                          # Interactive dashboard (Vercel)
├── reproduce.py                   # One-command reproduction (38 modules)
├── Dockerfile                     # Docker reproducibility package
├── requirements.txt               # Python dependencies
├── paper_english.md               # Full paper (35 sections)
└── paper_korean.md                # Full paper (Korean)
```

---

## 📈 Reproduction Pipeline

```
$ python reproduce.py
============================================================
  EthicaAI Reproduction Pipeline
  Phase: all  |  Mode: Full  |  Modules: 38
============================================================
  ✓ G1–G5: Core validation (convergence, sensitivity, cross-env)
  ✓ H1–H2: Evolutionary competition, mechanism decomposition
  ✓ M1–M4: Full sweep, mixed-SVO, communication, continuous
  ✓ N1–N4: MAPPO, partial obs, multi-resource, LLM
  ✓ O1–O8: Climate, vaccine, governance, Human-AI pilot
  ✓ P1–P6: Scale 1000, LMM, continuous, network, mechanism, adversarial
  ✓ Q2–Q6: Moran, moral theories, GNN, interpretability, policy

  Total: 38/38 succeeded
  Figures: 74 generated
  🎉 전체 재현 성공!
```

---

## 📜 Citation

```bibtex
@article{heo2026ethicaai,
  title={Beyond Homo Economicus: Computational Verification of Amartya Sen's
         Meta-Ranking Theory in Multi-Agent Social Dilemmas},
  author={Anonymous},
  journal={arXiv preprint arXiv:2602.XXXXX},
  year={2026},
  note={74 figures, 38 modules, 8 environments, NeurIPS 2026}
}
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

> *Built with ❤️ by the Antigravity Team.*
