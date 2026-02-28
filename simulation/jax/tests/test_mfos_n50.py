"""N=50 full comparison: M-FOS vs Meta-Ranking."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from simulation.jax.analysis.comparison_sota import run_mfos, run_meta_ranking

print("=" * 65)
print("  N=50 COMPARISON: M-FOS (Real) vs Meta-Ranking")
print("=" * 65)

# Benign condition
print("\n--- Benign (0% adversary) ---")
for seed in [42, 123]:
    mf = run_mfos(50, seed)
    mr = run_meta_ranking(50, seed)
    print(f"  Seed {seed:3d} | M-FOS  Coop:{mf['coop']:.3f} Welf:{mf['welfare']:.1f} Time:{mf['time_ms']:.0f}ms")
    print(f"  Seed {seed:3d} | MetaRk Coop:{mr['coop']:.3f} Welf:{mr['welfare']:.1f} Time:{mr['time_ms']:.0f}ms")

# Byzantine 50%
print("\n--- Byzantine 50% ---")
for seed in [42, 123]:
    mf = run_mfos(50, seed, 0.5)
    mr = run_meta_ranking(50, seed, 0.5)
    print(f"  Seed {seed:3d} | M-FOS  Coop:{mf['coop']:.3f} Welf:{mf['welfare']:.1f}")
    print(f"  Seed {seed:3d} | MetaRk Coop:{mr['coop']:.3f} Welf:{mr['welfare']:.1f}")

print("\n  DONE!")
