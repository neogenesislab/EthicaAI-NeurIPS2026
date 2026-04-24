"""Quick smoke test for M-FOS implementation."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from simulation.jax.analysis.comparison_sota import run_mfos, run_meta_ranking

print("=" * 60)
print("  M-FOS Smoke Test (N=20, seed=42)")
print("=" * 60)

# Test 1: Basic run
print("\n[1] M-FOS basic run (N=20, no adversary)...")
result = run_mfos(20, seed=42, adversarial_frac=0.0)
print(f"  Coop: {result['coop']:.3f}")
print(f"  Welfare: {result['welfare']:.1f}")
print(f"  Gini: {result['gini']:.4f}")
print(f"  Time: {result['time_ms']:.0f}ms")
print(f"  Time/step: {result['time_per_step_ms']:.3f}ms")

# Test 2: Adversarial run
print("\n[2] M-FOS adversarial run (N=20, 50% adversary)...")
result_adv = run_mfos(20, seed=42, adversarial_frac=0.5)
print(f"  Coop: {result_adv['coop']:.3f}")
print(f"  Welfare: {result_adv['welfare']:.1f}")

# Test 3: Compare with Meta-Ranking
print("\n[3] Meta-Ranking comparison (N=20, no adversary)...")
result_mr = run_meta_ranking(20, seed=42, adversarial_frac=0.0)
print(f"  Coop: {result_mr['coop']:.3f}")
print(f"  Welfare: {result_mr['welfare']:.1f}")

print("\n" + "=" * 60)
print("  COMPARISON")
print("=" * 60)
print(f"  {'':20s} {'M-FOS':>10s} {'Meta-Rank':>10s}")
print(f"  {'Cooperation':20s} {result['coop']:10.3f} {result_mr['coop']:10.3f}")
print(f"  {'Welfare':20s} {result['welfare']:10.1f} {result_mr['welfare']:10.1f}")
print(f"  {'Time (ms)':20s} {result['time_ms']:10.0f} {result_mr['time_ms']:10.0f}")
print("\n  SMOKE TEST PASSED!" if result['coop'] > 0 else "\n  !! SMOKE TEST FAILED !!")
