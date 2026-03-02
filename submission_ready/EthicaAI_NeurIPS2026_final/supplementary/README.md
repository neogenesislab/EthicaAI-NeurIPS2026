# EthicaAI: Supplementary Materials

## Quick Reproduction

```bash
pip install numpy matplotlib
python reproduce_quick.py        # Full reproduction (~15 min CPU)
python reproduce_quick.py --fast # Smoke test (~2 min)
```

## Scripts

| Script | Purpose | Paper Reference |
|--------|---------|----------------|
| `reproduce_quick.py` | One-command reproduction | All tables |
| `ppo_nash_trap.py` | Ind. REINFORCE Nash Trap | Table 3 (rows 1-3) |
| `p3_fast_baselines.py` | True IPPO, MAPPO, QMIX | Table 3 (rows 4-6) |
| `scale_test_n100.py` | N=100 scale test | Table 5 |
| `mappo_emergence.py` | Emergence experiment | Section 3.2 |
| `dnn_ablation.py` | DNN architecture ablation | Appendix B |
| `kpg_experiment.py` | K-level policy gradient | Appendix C |

## Dependencies

- Python 3.8+
- NumPy >= 1.20
- Matplotlib >= 3.5 (for plots only)
