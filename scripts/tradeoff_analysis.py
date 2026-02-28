"""
A-1: Trade-off Deep Analysis — φ₁=1.0 (Learned) vs φ₁=0.21 (Hand-crafted)

Three analyses for reviewer defense:
  1. Individual Survival Rate: Per-agent cumulative payoff CDF
  2. Adaptive Adversary: Exploitative Scarcity attack scenario
  3. Over-contribution: Marginal utility of moral commitment
"""

import numpy as np
import json
import os
import time

# ============================================================
# Constants
# ============================================================
N_AGENTS = 50
T_ROUNDS = 200
N_SEEDS = 10
MULTIPLIER = 1.6
ENDOWMENT = 20.0
ALPHA_EMA = 0.6


# ============================================================
# g functions: Learned vs Hand-crafted
# ============================================================
def g_learned(theta, R):
    """Learned g_φ: φ₁=1.0 (crisis base), φ₃=0.307, φ₄=1.652"""
    phi = [1.297, 1.000, 0.004, 0.307, 1.652, 0.405]
    svo = 1.0 / (1.0 + np.exp(-phi[0] * np.sin(theta)))
    if R < phi[3]:
        raw = phi[1] + phi[2] * R
    elif R > (1.0 - phi[3]):
        raw = phi[4] + phi[5] * R
    else:
        raw = phi[1] + (phi[4] - phi[1]) * (R - phi[3]) / max(1.0 - 2 * phi[3], 0.01)
    return float(np.clip(svo * raw, 0.0, 1.0))


def g_handcrafted(theta, R):
    """Hand-crafted g: φ₁=0.21 (crisis base = 0.3·sin(45°))"""
    base = np.sin(theta)
    if R < 0.2:
        return max(0.0, 0.3 * base)
    elif R > 0.7:
        return min(1.0, 1.5 * base)
    else:
        return float(np.clip(base * (0.7 + 1.6 * R), 0.0, 1.0))


# ============================================================
# PGG with per-agent tracking
# ============================================================
def run_pgg_detailed(g_func, n_agents=N_AGENTS, t_rounds=T_ROUNDS,
                     byz_frac=0.0, byz_type="fixed", seed=42):
    """PGG with detailed per-agent metrics.

    byz_type: "fixed" (always defect) or "exploitative" (drain then exploit)
    """
    rng = np.random.RandomState(seed)
    svo_angles = rng.uniform(np.radians(20), np.radians(70), n_agents)
    n_byz = int(n_agents * byz_frac)

    R_t = 0.5
    lambdas = np.array([g_func(svo_angles[i], R_t) for i in range(n_agents)])

    # Per-agent tracking
    cumulative_payoffs = np.zeros(n_agents)
    min_payoffs = np.full(n_agents, float('inf'))
    rounds_negative = np.zeros(n_agents, dtype=int)

    welfare_history = []
    resource_history = [R_t]
    contribution_history = []

    for t in range(t_rounds):
        contributions = np.zeros(n_agents)

        for i in range(n_agents):
            if i < n_byz:
                if byz_type == "exploitative":
                    # Adaptive: contribute nothing when R is low (drain),
                    # contribute minimally when R is high (blend in)
                    if R_t > 0.5:
                        contributions[i] = ENDOWMENT * 0.1  # Minimal to avoid detection
                    else:
                        contributions[i] = 0.0  # Full exploitation
                else:
                    contributions[i] = 0.0
            else:
                contributions[i] = ENDOWMENT * lambdas[i]

        # PGG payoffs
        total_contrib = contributions.sum()
        public_good = (total_contrib * MULTIPLIER) / n_agents
        payoffs = (ENDOWMENT - contributions) + public_good

        # Track per-agent metrics
        cumulative_payoffs += payoffs
        for i in range(n_agents):
            min_payoffs[i] = min(min_payoffs[i], payoffs[i])
            if payoffs[i] < ENDOWMENT * 0.5:  # Below half endowment = stressed
                rounds_negative[i] += 1

        welfare_history.append(float(payoffs.mean()))
        contribution_history.append(float(contributions.mean()))

        # Resource dynamics
        coop_ratio = np.mean(contributions) / ENDOWMENT
        R_t = np.clip(R_t + 0.1 * (coop_ratio - 0.4), 0.0, 1.0)
        resource_history.append(R_t)

        for i in range(n_byz, n_agents):
            target = g_func(svo_angles[i], R_t)
            lambdas[i] = ALPHA_EMA * lambdas[i] + (1 - ALPHA_EMA) * target

    return {
        "welfare": float(np.mean(welfare_history)),
        "cumulative_payoffs": cumulative_payoffs.tolist(),
        "min_payoffs": min_payoffs.tolist(),
        "rounds_stressed": rounds_negative.tolist(),
        "resource_history": [float(r) for r in resource_history],
        "contribution_history": contribution_history,
        "gini": float(gini_coefficient(cumulative_payoffs)),
        "bottom_10_pct": float(np.percentile(cumulative_payoffs, 10)),
        "bottom_10_pct_honest": float(np.percentile(cumulative_payoffs[int(n_agents*byz_frac):], 10)),
    }


def gini_coefficient(x):
    x = np.array(x, dtype=float)
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


# ============================================================
# Analysis 1: Individual Survival Rate
# ============================================================
def analysis_survival(seeds):
    """Compare per-agent payoff distributions."""
    print("\n" + "=" * 65)
    print("  Analysis 1: Individual Survival Rate (φ₁=1.0 vs φ₁=0.21)")
    print("=" * 65)

    results = {}
    for label, g_func in [("learned", g_learned), ("handcrafted", g_handcrafted)]:
        all_cum = []
        all_bottom10 = []
        all_stressed = []
        all_gini = []

        for byz in [0.0, 0.3, 0.5]:
            bf_key = f"byz_{int(byz*100)}"
            cum_list = []
            bottom10_list = []
            stressed_list = []
            gini_list = []

            for seed in seeds:
                r = run_pgg_detailed(g_func, byz_frac=byz, seed=seed)
                cum_list.append(np.mean(r["cumulative_payoffs"]))
                bottom10_list.append(r["bottom_10_pct_honest"])
                stressed_list.append(np.mean(r["rounds_stressed"]))
                gini_list.append(r["gini"])

            results.setdefault(label, {})[bf_key] = {
                "avg_cumulative": float(np.mean(cum_list)),
                "bottom_10_pct": float(np.mean(bottom10_list)),
                "avg_stressed_rounds": float(np.mean(stressed_list)),
                "gini": float(np.mean(gini_list)),
            }

    # Print comparison
    for byz in [0.0, 0.3, 0.5]:
        bf_key = f"byz_{int(byz*100)}"
        l = results["learned"][bf_key]
        h = results["handcrafted"][bf_key]
        print(f"\n  Byz={byz*100:.0f}%:")
        print(f"    {'Metric':>22} | {'Learned':>10} | {'Handcraft':>10} | {'Δ':>8}")
        print(f"    {'-'*55}")
        for metric in ["avg_cumulative", "bottom_10_pct", "avg_stressed_rounds", "gini"]:
            d = l[metric] - h[metric]
            print(f"    {metric:>22} | {l[metric]:10.2f} | {h[metric]:10.2f} | {d:+8.2f}")

    return results


# ============================================================
# Analysis 2: Adaptive Adversary (Exploitative Scarcity)
# ============================================================
def analysis_adaptive_adversary(seeds):
    """Test against adversaries that exploit the λ_t pattern."""
    print("\n" + "=" * 65)
    print("  Analysis 2: Adaptive Adversary - Exploitative Scarcity")
    print("=" * 65)

    results = {}
    for label, g_func in [("learned", g_learned), ("handcrafted", g_handcrafted)]:
        for byz_type in ["fixed", "exploitative"]:
            key = f"{label}_{byz_type}"
            welfare_list = []
            coop_list = []
            resource_list = []

            for seed in seeds:
                r = run_pgg_detailed(g_func, byz_frac=0.3, byz_type=byz_type, seed=seed)
                welfare_list.append(r["welfare"])
                resource_list.append(np.mean(r["resource_history"]))

            results[key] = {
                "welfare": float(np.mean(welfare_list)),
                "welfare_se": float(np.std(welfare_list) / np.sqrt(len(seeds))),
                "avg_resource": float(np.mean(resource_list)),
            }

    print(f"\n  {'Config':>28} | {'Welfare':>10} | {'±SE':>6} | {'Avg R':>8}")
    print(f"  {'-'*60}")
    for key in sorted(results):
        r = results[key]
        print(f"  {key:>28} | {r['welfare']:10.2f} | {r['welfare_se']:6.2f} | {r['avg_resource']:8.3f}")

    # Compute resilience: exploitative vs fixed
    for label in ["learned", "handcrafted"]:
        fixed = results[f"{label}_fixed"]["welfare"]
        exploit = results[f"{label}_exploitative"]["welfare"]
        drop = (fixed - exploit) / fixed * 100
        print(f"\n  {label}: Exploitative drop = {drop:+.2f}%")

    return results


# ============================================================
# Analysis 3: Marginal Utility of Commitment
# ============================================================
def analysis_marginal_utility(seeds):
    """Test welfare at different φ₁ values to find the optimal crisis commitment."""
    print("\n" + "=" * 65)
    print("  Analysis 3: Marginal Utility of Crisis Commitment (φ₁ sweep)")
    print("=" * 65)

    phi1_values = [0.0, 0.1, 0.21, 0.3, 0.5, 0.7, 0.85, 1.0]
    results = {}

    for phi1 in phi1_values:
        def g_sweep(theta, R, _phi1=phi1):
            phi = [1.297, _phi1, 0.004, 0.307, 1.652, 0.405]
            svo = 1.0 / (1.0 + np.exp(-phi[0] * np.sin(theta)))
            if R < phi[3]:
                raw = phi[1] + phi[2] * R
            elif R > (1.0 - phi[3]):
                raw = phi[4] + phi[5] * R
            else:
                raw = phi[1] + (phi[4] - phi[1]) * (R - phi[3]) / max(1.0 - 2 * phi[3], 0.01)
            return float(np.clip(svo * raw, 0.0, 1.0))

        welfare_benign = []
        welfare_byz30 = []
        welfare_byz50 = []
        bottom10_byz30 = []

        for seed in seeds:
            r0 = run_pgg_detailed(g_sweep, byz_frac=0.0, seed=seed)
            r3 = run_pgg_detailed(g_sweep, byz_frac=0.3, seed=seed)
            r5 = run_pgg_detailed(g_sweep, byz_frac=0.5, seed=seed)
            welfare_benign.append(r0["welfare"])
            welfare_byz30.append(r3["welfare"])
            welfare_byz50.append(r5["welfare"])
            bottom10_byz30.append(r3["bottom_10_pct_honest"])

        results[str(phi1)] = {
            "phi1": phi1,
            "welfare_benign": float(np.mean(welfare_benign)),
            "welfare_byz30": float(np.mean(welfare_byz30)),
            "welfare_byz50": float(np.mean(welfare_byz50)),
            "bottom10_byz30": float(np.mean(bottom10_byz30)),
        }

    print(f"\n  {'φ₁':>6} | {'W(0%)':>8} | {'W(30%)':>8} | {'W(50%)':>8} | {'Bot10(30%)':>10}")
    print(f"  {'-'*52}")
    for phi1 in phi1_values:
        r = results[str(phi1)]
        print(f"  {phi1:6.2f} | {r['welfare_benign']:8.2f} | {r['welfare_byz30']:8.2f} | "
              f"{r['welfare_byz50']:8.2f} | {r['bottom10_byz30']:10.1f}")

    # Find optimal φ₁ per condition
    for cond in ["welfare_benign", "welfare_byz30", "welfare_byz50"]:
        best = max(results.values(), key=lambda x: x[cond])
        print(f"\n  Optimal φ₁ for {cond}: {best['phi1']:.2f} (W={best[cond]:.2f})")

    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'tradeoff_analysis')
    os.makedirs(OUT, exist_ok=True)

    seeds = list(range(N_SEEDS))
    t0 = time.time()

    print("=" * 65)
    print("  A-1: Trade-off Deep Analysis")
    print("  φ₁=1.0 (Learned/Resilient) vs φ₁=0.21 (Handcrafted/Safe)")
    print("=" * 65)

    # Run all 3 analyses
    survival = analysis_survival(seeds)
    adversary = analysis_adaptive_adversary(seeds)
    marginal = analysis_marginal_utility(seeds)

    total_time = time.time() - t0

    # Save
    output = {
        "time_seconds": float(total_time),
        "n_seeds": N_SEEDS,
        "survival_analysis": survival,
        "adaptive_adversary": adversary,
        "marginal_utility_sweep": marginal,
    }

    path = os.path.join(OUT, "tradeoff_results.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Results: {path}")
    print("\n  A-1 COMPLETE!")
