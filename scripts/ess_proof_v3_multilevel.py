"""
ESS Proof v3 — Multi-Level Selection / Price Equation Decomposition
Phase 1, Final Frame: Group-Level Evolutionary Stability

Proposition 3 (Final):
  Individual-level ESS is mathematically impossible in linear PGG
  (free-riding always dominates in per-step payoff).
  However, under group selection dynamics (population replacement),
  Situational Commitment achieves GROUP-LEVEL evolutionary stability:
  Situational-dominant groups produce 58% higher collective fitness
  than Defector-dominant groups, driving evolutionary dominance.

Price Equation Decomposition:
  Δp = Cov(w_g, p_g)/w̄  +  E[Cov_within(w_i, p_i)]/w̄
       ├── Between-group ──┤  ├── Within-group ──────────┤
       (favors cooperation)     (favors defection)

  Group selection wins when |Between| > |Within|
"""

import numpy as np
import json
import os

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Environment (same as paper)
ENDOWMENT = 100
MPCR = 1.6
ALPHA = 0.02
THRESHOLD = 0.3
R_CRISIS = 0.2
R_ABUNDANCE = 0.7
SVO_THETA = np.radians(45)
T_EPISODE = 200


def lambda_sit(R_t):
    """Situational Commitment λ(R_t)."""
    base = np.sin(SVO_THETA)
    if R_t < R_CRISIS:
        return max(0.0, 0.3 * base)
    elif R_t > R_ABUNDANCE:
        return min(1.0, 1.5 * base)
    else:
        return base * (0.7 + 1.6 * R_t)


def run_group(n, x_sit, seed=0):
    """Run one group of n agents for T_EPISODE steps.
    Returns: (mean_payoff_sit, mean_payoff_def, group_mean_payoff, R_final)
    """
    rng = np.random.RandomState(seed)
    n_s = max(int(round(n * x_sit)), 0)
    n_d = n - n_s
    R = 0.5
    pays_s, pays_d = [], []

    for _ in range(T_EPISODE):
        lam = lambda_sit(R)
        cs = np.clip(lam * ENDOWMENT + rng.normal(0, 3, n_s), 0, ENDOWMENT) if n_s > 0 else np.array([])
        cd = np.clip(rng.normal(0, 3, n_d), 0, ENDOWMENT) if n_d > 0 else np.array([])
        allc = np.concatenate([cs, cd])
        pub = allc.sum() * MPCR / n
        if n_s > 0:
            pays_s.append(float((ENDOWMENT - cs + pub).mean()))
        if n_d > 0:
            pays_d.append(float((ENDOWMENT - cd + pub).mean()))
        R = np.clip(R + ALPHA * (allc.mean() / ENDOWMENT - THRESHOLD), 0, 1)

    fs = float(np.mean(pays_s)) if pays_s else 0.0
    fd = float(np.mean(pays_d)) if pays_d else 0.0
    fg = (fs * n_s + fd * n_d) / n if n > 0 else 0.0
    return fs, fd, fg, R


# ============================================================
# Test 1: Group Fitness Landscape
# ============================================================
def group_fitness_landscape(n=50, n_seeds=5):
    """Group-mean fitness as function of Situational fraction x."""
    xs = np.linspace(0, 1, 21)
    results = []
    for x in xs:
        fgs, fss, fds = [], [], []
        for s in range(n_seeds):
            fs, fd, fg, _ = run_group(n, x, seed=s)
            fgs.append(fg)
            fss.append(fs)
            fds.append(fd)
        results.append({
            "x": float(x),
            "group_fitness": float(np.mean(fgs)),
            "sit_payoff": float(np.mean(fss)),
            "def_payoff": float(np.mean(fds)),
            "within_disadvantage": float(np.mean(fds)) - float(np.mean(fss)),
        })
    return results


# ============================================================
# Test 2: Price Equation Decomposition
# ============================================================
def price_equation(n_groups=20, n_per_group=50, n_seeds=3):
    """Decompose selection into between-group and within-group components.

    Simulate n_groups groups with varying Situational fraction,
    then compute Price equation components.
    """
    # Groups with different compositions (representing population structure)
    x_fracs = np.linspace(0, 1, n_groups)
    results_all_seeds = []

    for seed in range(n_seeds):
        group_data = []
        for i, x in enumerate(x_fracs):
            fs, fd, fg, _ = run_group(n_per_group, x, seed=seed * 100 + i)
            n_s = int(round(n_per_group * x))
            n_d = n_per_group - n_s
            group_data.append({
                "x": float(x), "fg": fg, "fs": fs, "fd": fd,
                "n_s": n_s, "n_d": n_d,
            })

        # Price equation: Δp̄ = Cov(w_g, p_g)/w̄ + E[Cov_within]/w̄
        # p_g = fraction of Situational in group g
        # w_g = group fitness (relative)
        p_g = np.array([g["x"] for g in group_data])
        w_g = np.array([g["fg"] for g in group_data])
        w_bar = w_g.mean()

        # Between-group: Cov(w_g, p_g) / w̄
        between = np.cov(w_g, p_g)[0, 1] / w_bar if w_bar > 0 else 0

        # Within-group: E[Cov_within(w_i, p_i)] / w̄
        # For each group, within-group covariance between individual fitness & type
        within_covs = []
        for g in group_data:
            if g["n_s"] > 0 and g["n_d"] > 0:
                # Individual fitnesses: n_s agents with fs, n_d with fd
                w_i = np.array([g["fs"]] * g["n_s"] + [g["fd"]] * g["n_d"])
                p_i = np.array([1.0] * g["n_s"] + [0.0] * g["n_d"])
                cov_w = np.cov(w_i, p_i)[0, 1] if len(w_i) > 1 else 0.0
                within_covs.append(cov_w)
            else:
                within_covs.append(0.0)
        within = np.mean(within_covs) / w_bar if w_bar > 0 else 0

        delta_p = between + within

        results_all_seeds.append({
            "between_group": float(between),
            "within_group": float(within),
            "delta_p": float(delta_p),
            "group_selection_dominates": abs(between) > abs(within),
        })

    # Average across seeds
    avg = {
        "between_group": float(np.mean([r["between_group"] for r in results_all_seeds])),
        "within_group": float(np.mean([r["within_group"] for r in results_all_seeds])),
        "delta_p": float(np.mean([r["delta_p"] for r in results_all_seeds])),
        "between_se": float(np.std([r["between_group"] for r in results_all_seeds]) / np.sqrt(n_seeds)),
        "within_se": float(np.std([r["within_group"] for r in results_all_seeds]) / np.sqrt(n_seeds)),
        "group_selection_wins": abs(float(np.mean([r["between_group"] for r in results_all_seeds]))) >
                                abs(float(np.mean([r["within_group"] for r in results_all_seeds]))),
        "ratio_between_to_within": abs(float(np.mean([r["between_group"] for r in results_all_seeds]))) /
                                   max(abs(float(np.mean([r["within_group"] for r in results_all_seeds]))), 1e-10),
    }
    return avg, results_all_seeds


# ============================================================
# Test 3: Evolutionary Tournament Replication (mini)
# ============================================================
def evolutionary_tournament(n_per_group=50, n_generations=100, n_groups=10, seed=42):
    """Mini evolutionary tournament: groups compete, worst-performing
    groups are replaced by copies of best-performing groups.

    Tracks Situational fraction over generations.
    """
    rng = np.random.RandomState(seed)
    # Initialize: uniform random Situational fractions
    x_fracs = rng.uniform(0, 1, n_groups)

    history = [{"gen": 0, "x_mean": float(x_fracs.mean()), "x_std": float(x_fracs.std())}]

    for gen in range(1, n_generations + 1):
        # Run each group
        group_fitness = []
        for i in range(n_groups):
            _, _, fg, _ = run_group(n_per_group, x_fracs[i], seed=seed * 1000 + gen * 100 + i)
            group_fitness.append(fg)

        group_fitness = np.array(group_fitness)

        # Selection: replace worst group with copy of best
        worst = np.argmin(group_fitness)
        best = np.argmax(group_fitness)
        x_fracs[worst] = x_fracs[best]

        # Mutation: small perturbation to all groups
        x_fracs += rng.normal(0, 0.02, n_groups)
        x_fracs = np.clip(x_fracs, 0, 1)

        if gen % 10 == 0 or gen == 1:
            history.append({
                "gen": gen,
                "x_mean": float(x_fracs.mean()),
                "x_std": float(x_fracs.std()),
                "best_fitness": float(group_fitness.max()),
                "worst_fitness": float(group_fitness.min()),
            })

    return {
        "final_x_mean": float(x_fracs.mean()),
        "converged_to_sit": float(x_fracs.mean()) > 0.8,
        "history": history,
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'ess_proof')
    os.makedirs(OUT, exist_ok=True)

    print("=" * 70)
    print("  Phase 1 v3: Multi-Level Selection ESS Proof")
    print("  Price Equation + Group Selection Dynamics")
    print("=" * 70)

    # --- Test 1: Group Fitness Landscape ---
    print("\n[Test 1] Group Fitness Landscape (N=50)")
    landscape = group_fitness_landscape(50, n_seeds=5)
    print(f"\n  {'x_Sit':>6s} {'F_group':>8s} {'F_sit':>8s} {'F_def':>8s} {'Within Δ':>10s}")
    print("  " + "-" * 48)
    for r in landscape:
        w = f"{r['within_disadvantage']:+.1f}" if r['def_payoff'] > 0 and r['sit_payoff'] > 0 else "N/A"
        print(f"  {r['x']:6.2f} {r['group_fitness']:8.1f} {r['sit_payoff']:8.1f} "
              f"{r['def_payoff']:8.1f} {w:>10s}")

    # Key comparison
    fg_full_sit = next(r['group_fitness'] for r in landscape if r['x'] == 1.0)
    fg_full_def = next(r['group_fitness'] for r in landscape if r['x'] == 0.0)
    print(f"\n  ★ Group fitness: All-Sit={fg_full_sit:.1f} vs All-Def={fg_full_def:.1f}")
    print(f"    Situational group advantage: {(fg_full_sit/fg_full_def - 1)*100:.1f}%")

    # --- Test 2: Price Equation ---
    print("\n\n[Test 2] Price Equation Decomposition (20 groups × 50 agents)")
    price_avg, price_seeds = price_equation(n_groups=20, n_per_group=50, n_seeds=5)

    print(f"\n  Between-group (favors Sit): {price_avg['between_group']:+.6f} ± {price_avg['between_se']:.6f}")
    print(f"  Within-group (favors Def):  {price_avg['within_group']:+.6f} ± {price_avg['within_se']:.6f}")
    print(f"  Δp̄ (net change):            {price_avg['delta_p']:+.6f}")
    print(f"  |Between|/|Within| ratio:   {price_avg['ratio_between_to_within']:.4f}x")
    print(f"  Group selection dominates:   {'YES ✓' if price_avg['group_selection_wins'] else 'NO ✗'}")

    # --- Test 3: Evolutionary Tournament ---
    print("\n\n[Test 3] Mini Evolutionary Tournament (10 groups, 100 generations)")
    tourney = evolutionary_tournament(n_per_group=50, n_generations=100, n_groups=10)

    print(f"\n  Final Situational fraction: {tourney['final_x_mean']:.3f}")
    print(f"  Converged to Situational: {'YES ✓' if tourney['converged_to_sit'] else 'NO ✗'}")
    print("\n  Generation trajectory:")
    for h in tourney["history"]:
        bar = "█" * int(h["x_mean"] * 30)
        print(f"  Gen {h['gen']:4d} | x̄={h['x_mean']:.3f} | {bar}")

    # --- Save ---
    results = {
        "framework": "Multi-Level Selection / Price Equation",
        "proposition": "Situational Commitment is a Group-Level ESS: not individually stable, but group-selected",
        "individual_ess_possible": False,
        "individual_ess_reason": "In linear PGG, free-riding always yields higher per-step payoff by exactly c_S (contribution saved)",
        "group_fitness_landscape": landscape,
        "price_equation": price_avg,
        "price_equation_per_seed": price_seeds,
        "evolutionary_tournament": tourney,
        "key_numbers": {
            "group_fitness_all_sit": fg_full_sit,
            "group_fitness_all_def": fg_full_def,
            "group_advantage_pct": float((fg_full_sit / fg_full_def - 1) * 100),
            "between_group_selection": price_avg["between_group"],
            "within_group_selection": price_avg["within_group"],
            "group_selection_ratio": price_avg["ratio_between_to_within"],
        },
    }

    path = os.path.join(OUT, "ess_proof_v3_multilevel.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NpEncoder)

    print(f"\n\n  Results: {path}")
    print("\n" + "=" * 70)
    print("  PROPOSITION 3 SUMMARY")
    print("=" * 70)
    print(f"  Individual ESS: IMPOSSIBLE (PGG structural property)")
    print(f"  Group fitness advantage: {(fg_full_sit/fg_full_def-1)*100:.1f}% (Sit > Def)")
    print(f"  Price Eq. |Between/Within|: {price_avg['ratio_between_to_within']:.4f}x")
    print(f"  Tournament convergence: x̄ = {tourney['final_x_mean']:.3f}")
    print(f"\n  Phase 1 v3 COMPLETE!")
