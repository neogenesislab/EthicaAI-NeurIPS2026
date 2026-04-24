"""
Full M-FOS vs Meta-Ranking Benchmark
10 seeds × 4 adversarial conditions (0%, 10%, 30%, 50%)
N=50 PGG environment, SVO=45° (Prosocial)

Outputs: JSON results + console table
"""
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from simulation.jax.analysis.comparison_sota import (
    run_mfos, run_meta_ranking, run_lola, run_lopt
)

# Configuration
N_AGENTS = 50
SEEDS = [0, 42, 123, 256, 999, 1337, 2024, 3141, 4269, 5555]
BYZ_FRACS = [0.0, 0.1, 0.3, 0.5]
ALGORITHMS = {
    "Meta-Ranking": run_meta_ranking,
    "M-FOS (Real)": run_mfos,
    "LOLA": run_lola,
    "LOPT": run_lopt,
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'mfos_benchmark')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 75)
print("  FULL BENCHMARK: M-FOS (Real) vs Meta-Ranking vs LOLA vs LOPT")
print(f"  N={N_AGENTS} | Seeds={len(SEEDS)} | Byz conditions={len(BYZ_FRACS)}")
print(f"  Total runs: {len(ALGORITHMS)} × {len(SEEDS)} × {len(BYZ_FRACS)} = "
      f"{len(ALGORITHMS) * len(SEEDS) * len(BYZ_FRACS)}")
print("=" * 75)

all_results = {}
t_global = time.perf_counter()

for algo_name, algo_fn in ALGORITHMS.items():
    all_results[algo_name] = {}

    for byz_frac in BYZ_FRACS:
        byz_key = f"byz_{int(byz_frac*100)}pct"
        seed_results = []

        for i, seed in enumerate(SEEDS):
            t0 = time.perf_counter()
            result = algo_fn(N_AGENTS, seed, adversarial_frac=byz_frac)
            dt = time.perf_counter() - t0

            seed_results.append(result)
            progress = f"[{i+1}/{len(SEEDS)}]"
            print(f"  {algo_name:20s} | Byz={int(byz_frac*100):2d}% | Seed={seed:4d} {progress} | "
                  f"Coop:{result['coop']:.3f} Welf:{result['welfare']:.1f} | {dt:.1f}s")

        # Aggregate
        agg = {
            "coop_mean": float(np.mean([r["coop"] for r in seed_results])),
            "coop_std": float(np.std([r["coop"] for r in seed_results])),
            "coop_se": float(np.std([r["coop"] for r in seed_results]) / np.sqrt(len(SEEDS))),
            "welfare_mean": float(np.mean([r["welfare"] for r in seed_results])),
            "welfare_std": float(np.std([r["welfare"] for r in seed_results])),
            "welfare_se": float(np.std([r["welfare"] for r in seed_results]) / np.sqrt(len(SEEDS))),
            "gini_mean": float(np.mean([r["gini"] for r in seed_results])),
            "gini_std": float(np.std([r["gini"] for r in seed_results])),
            "time_ms_mean": float(np.mean([r["time_ms"] for r in seed_results])),
            "raw_seeds": seed_results,
        }
        all_results[algo_name][byz_key] = agg
        print(f"  >>> {algo_name:20s} | Byz={int(byz_frac*100):2d}% AGGREGATE: "
              f"Coop={agg['coop_mean']:.3f}±{agg['coop_se']:.3f} | "
              f"Welfare={agg['welfare_mean']:.1f}±{agg['welfare_se']:.1f}\n")

elapsed_total = time.perf_counter() - t_global

# Save JSON
json_path = os.path.join(OUTPUT_DIR, "mfos_benchmark_results.json")
# Remove raw seeds for cleaner output
clean_results = {}
for algo, byz_data in all_results.items():
    clean_results[algo] = {}
    for byz_key, agg in byz_data.items():
        clean_results[algo][byz_key] = {k: v for k, v in agg.items() if k != "raw_seeds"}

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(clean_results, f, indent=2, ensure_ascii=False)

# Summary Table
print("\n" + "=" * 75)
print("  FINAL SUMMARY TABLE (10-seed averaged)")
print("=" * 75)
print(f"\n  {'Algorithm':20s} | {'Byz%':>4s} | {'Coop':>12s} | {'Welfare':>14s} | {'Time(ms)':>10s}")
print("  " + "-" * 70)

for algo_name in ALGORITHMS:
    for byz_frac in BYZ_FRACS:
        byz_key = f"byz_{int(byz_frac*100)}pct"
        a = all_results[algo_name][byz_key]
        marker = " ★" if algo_name == "Meta-Ranking" else ""
        print(f"  {algo_name:20s} | {int(byz_frac*100):3d}% | "
              f"{a['coop_mean']:.3f}±{a['coop_se']:.3f} | "
              f"{a['welfare_mean']:.1f}±{a['welfare_se']:.1f} | "
              f"{a['time_ms_mean']:10.0f}{marker}")
    print()

# Key comparisons
print("\n  KEY COMPARISONS (Benign → Byz-50%):")
for algo_name in ALGORITHMS:
    c0 = all_results[algo_name]["byz_0pct"]["coop_mean"]
    c50 = all_results[algo_name]["byz_50pct"]["coop_mean"]
    resilience = c50 / max(c0, 1e-8)
    w0 = all_results[algo_name]["byz_0pct"]["welfare_mean"]
    w50 = all_results[algo_name]["byz_50pct"]["welfare_mean"]
    print(f"  {algo_name:20s} | Coop: {c0:.3f}→{c50:.3f} (resilience={resilience:.2f}) | "
          f"Welfare: {w0:.1f}→{w50:.1f}")

print(f"\n  Total benchmark time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
print(f"  Results saved: {json_path}")
print("\n  BENCHMARK COMPLETE!")
