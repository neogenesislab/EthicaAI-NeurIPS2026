# EthicaAI — Supplementary Material

## Reproducibility Package

This archive contains the core simulation scripts for reproducing all experiments in the paper
"From Situational to Unconditional: The Spectrum of Moral Commitment Required for
Multi-Agent Survival in Non-linear Social Dilemmas."

### Requirements

- Python 3.10+
- NumPy, Matplotlib

### Scripts

| File | Description |
|------|-------------|
| `mappo_emergence.py` | REINFORCE agent training in linear & non-linear PGG (Figs 3-4, Tables 1-3) |
| `nonlinear_pgg.py` | Non-linear PGG environment with tipping point dynamics (Table 4) |
| `spatial_dilemma.py` | Spatial social dilemma on grid (Table 5) |
| `phase_diagram.py` | R_crit × φ₁ phase diagram sweep (Fig 5) |
| `melting_pot_comparison.py` | Multi-axis benchmark comparison (Appendix H) |
| `reproduce.py` | One-click full reproduction pipeline |

### Quick Start

```bash
pip install numpy matplotlib
python reproduce.py
```

### Anonymous Repository

Full code with environment configs: <https://anonymous.4open.science/r/EthicaAI/>
