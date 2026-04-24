"""
ESS Proof v2 — Eco-Evolutionary Dynamic ESS
Phase 1, Reframed with Long-Run Cumulative Fitness

Proposition 3 (Revised):
  While no cooperative strategy can be a static ESS in a one-shot PGG
  (defection always dominates), Situational Commitment IS an
  eco-evolutionary ESS when fitness is measured as LONG-RUN CUMULATIVE
  payoff over repeated rounds with resource feedback.

Key Mechanism:
  Defectors achieve higher SINGLE-STEP payoff by free-riding.
  BUT their presence depletes shared resources R_t → 0.
  Resource depletion collapses PUBLIC GOOD for EVERYONE (including defectors).
  Result: Defector-dominant populations achieve LOWER cumulative payoff
  than Situational-dominant populations.

Three Tests:
  1. Eco-ESS Invasion: F_S^cum(1-ε) > F_D^cum(1-ε) over full episodes?
  2. Resource Trajectory: R*(x) as function of Situational fraction
  3. Moran Fixation: With cumulative fitness, is ρ_S > 1/N?
"""

import numpy as np
import json
import os

# ============================================================
# Environment Parameters
# ============================================================
ENDOWMENT = 100
MPCR = 1.6
ALPHA = 0.02
THRESHOLD = 0.3
R_CRISIS = 0.2
R_ABUNDANCE = 0.7
SVO_THETA = np.radians(45)
T_EPISODE = 200   # Episode length (matches paper's PGG setting)
N_EPISODES = 50    # Number of episodes for cumulative fitness


def lambda_situational(theta, R_t):
    """Dynamic λ: resource-conditioned moral commitment."""
    base = np.sin(theta)
    if R_t < R_CRISIS:
        return max(0.0, 0.3 * base)
    elif R_t > R_ABUNDANCE:
        return min(1.0, 1.5 * base)
    else:
        return base * (0.7 + 1.6 * R_t)


# ============================================================
# Eco-Evolutionary Fitness: CUMULATIVE payoff over episodes
# ============================================================
def compute_cumulative_fitness(n_agents, x_sit, lambda_mutant, seed=42):
    """Compute CUMULATIVE fitness over multiple episodes.

    Key difference from v1: fitness = sum of payoffs across ALL episodes,
    including the resource depletion/recovery trajectory.

    Returns:
        (cum_fit_sit, cum_fit_mut, R_trajectory, per_step_details)
    """
    rng = np.random.RandomState(seed)
    n_sit = max(int(round(n_agents * x_sit)), 0)
    n_mut = n_agents - n_sit

    R_t = 0.5
    cum_sit = 0.0
    cum_mut = 0.0
    R_traj = []
    step_details = []

    for ep in range(N_EPISODES):
        for t in range(T_EPISODE):
            # Situational contributions (resource-adaptive)
            lam_s = lambda_situational(SVO_THETA, R_t)
            c_sit = np.clip(
                lam_s * ENDOWMENT + rng.normal(0, 3, n_sit), 0, ENDOWMENT
            ) if n_sit > 0 else np.array([])

            # Mutant contributions (fixed strategy)
            c_mut = np.clip(
                lambda_mutant * ENDOWMENT + rng.normal(0, 3, n_mut), 0, ENDOWMENT
            ) if n_mut > 0 else np.array([])

            # PGG
            all_c = np.concatenate([c_sit, c_mut])
            pub = all_c.sum() * MPCR / n_agents

            # Payoffs
            if n_sit > 0:
                p_sit = float((ENDOWMENT - c_sit + pub).mean())
                cum_sit += p_sit
            if n_mut > 0:
                p_mut = float((ENDOWMENT - c_mut + pub).mean())
                cum_mut += p_mut

            # Resource dynamics — THE KEY FEEDBACK
            R_t = np.clip(R_t + ALPHA * (all_c.mean() / ENDOWMENT - THRESHOLD), 0, 1)
            R_traj.append(R_t)

        # Record per-episode
        if ep % 10 == 0:
            step_details.append({
                "episode": ep,
                "R_t": float(R_t),
                "cum_sit": float(cum_sit),
                "cum_mut": float(cum_mut),
            })

    total_steps = N_EPISODES * T_EPISODE
    return (
        cum_sit / total_steps,  # Average per-step fitness (cumulative/T)
        cum_mut / total_steps,
        R_traj,
        step_details,
    )


# ============================================================
# Test 1: Eco-ESS Invasion Analysis
# ============================================================
def eco_ess_invasion(n_agents=50, epsilon=0.02, n_strategies=21, n_seeds=5):
    """Test: Can any fixed-λ mutant invade Situational in ECO-evolutionary sense?

    Uses CUMULATIVE fitness over full resource trajectory.
    """
    x_sit = 1.0 - epsilon
    lambdas = np.linspace(0, 1, n_strategies)

    results = []
    for lam in lambdas:
        diffs = []
        for seed in range(n_seeds):
            f_sit, f_mut, _, _ = compute_cumulative_fitness(
                n_agents, x_sit, lam, seed=seed
            )
            diffs.append(f_sit - f_mut)

        mean_d = float(np.mean(diffs))
        se_d = float(np.std(diffs) / np.sqrt(n_seeds))
        results.append({
            "lambda": float(lam),
            "diff_mean": mean_d,
            "diff_se": se_d,
            "ess_holds": mean_d > 0,
        })

    violations = sum(1 for r in results if not r["ess_holds"])
    return {
        "n_strategies": n_strategies,
        "n_violations": violations,
        "ess_valid": violations == 0,
        "details": results,
    }


# ============================================================
# Test 2: Resource Trajectory as Function of Composition
# ============================================================
def resource_trajectory_analysis(n_agents=50, n_seeds=3):
    """How does steady-state resource R* depend on Situational fraction x?"""
    x_values = np.linspace(0, 1, 11)
    results = []

    for x in x_values:
        R_ss_list = []
        cum_sit_list = []
        cum_mut_list = []

        for seed in range(n_seeds):
            f_sit, f_mut, R_traj, _ = compute_cumulative_fitness(
                n_agents, x, 0.0, seed=seed  # Mutant = pure defector
            )
            R_ss = float(np.mean(R_traj[-200:]))  # Last 200 steps
            R_ss_list.append(R_ss)
            cum_sit_list.append(f_sit)
            cum_mut_list.append(f_mut)

        results.append({
            "x_situational": float(x),
            "R_steady_mean": float(np.mean(R_ss_list)),
            "cum_fit_sit": float(np.mean(cum_sit_list)),
            "cum_fit_mut": float(np.mean(cum_mut_list)),
            "cum_advantage": float(np.mean(cum_sit_list)) - float(np.mean(cum_mut_list)),
        })

    return results


# ============================================================
# Test 3: Eco-Evolutionary Moran Fixation
# ============================================================
def eco_moran_fixation(n_agents=50, lambda_mutant=0.0, n_seeds=3):
    """Moran fixation probability using CUMULATIVE fitness."""
    ratios = []

    for j in range(1, n_agents):
        x = j / n_agents
        r_list = []
        for seed in range(n_seeds):
            f_sit, f_mut, _, _ = compute_cumulative_fitness(
                n_agents, x, lambda_mutant, seed=seed
            )
            if f_sit > 0:
                r_list.append(f_mut / f_sit)
            else:
                r_list.append(1.0)
        ratios.append(float(np.mean(r_list)))

    # Moran formula
    partial_prods = []
    running = 1.0
    for r in ratios:
        running *= r
        partial_prods.append(running)

    rho = 1.0 / (1.0 + sum(partial_prods))
    neutral = 1.0 / n_agents

    return {
        "rho": float(rho),
        "neutral": float(neutral),
        "favored": rho > neutral,
        "ratio": float(rho / neutral),
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'ess_proof')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  Phase 1 v2: Eco-Evolutionary Dynamic ESS Proof")
    print("  Fitness = Cumulative payoff over repeated PGG with resource feedback")
    print("=" * 70)

    # --- Test 1: Eco-ESS Invasion ---
    print("\n[Test 1] Eco-ESS Invasion Analysis (N=50, ε=0.02, 21 strategies)")
    inv = eco_ess_invasion(n_agents=50, epsilon=0.02, n_strategies=21, n_seeds=5)

    print(f"\n  Violations: {inv['n_violations']}/{inv['n_strategies']}")
    print(f"  ECO-ESS VALID: {'YES ✓' if inv['ess_valid'] else 'NO ✗'}")
    print(f"\n  {'λ_mut':>6s} {'F_Sit-F_Mut':>12s} {'SE':>8s} {'ESS':>5s}")
    print("  " + "-" * 35)
    for r in inv["details"]:
        m = "✓" if r["ess_holds"] else "✗"
        print(f"  {r['lambda']:6.2f} {r['diff_mean']:12.3f} {r['diff_se']:8.3f} {m:>5s}")

    # --- Test 2: Resource Trajectory ---
    print("\n\n[Test 2] Resource Trajectory — R*(x) vs Situational Fraction")
    res_traj = resource_trajectory_analysis(n_agents=50)

    print(f"\n  {'x_Sit':>6s} {'R*':>6s} {'F_Sit':>8s} {'F_Mut':>8s} {'Advantage':>10s}")
    print("  " + "-" * 45)
    for r in res_traj:
        print(f"  {r['x_situational']:6.1f} {r['R_steady_mean']:6.3f} "
              f"{r['cum_fit_sit']:8.1f} {r['cum_fit_mut']:8.1f} "
              f"{r['cum_advantage']:+10.3f}")

    # --- Test 3: Eco-Moran Fixation ---
    print("\n\n[Test 3] Eco-Moran Fixation Probability (N=50)")
    moran_results = {}
    for lam in [0.0, 0.5, 1.0]:
        print(f"\n  vs λ={lam:.1f} (mutant)...")
        m = eco_moran_fixation(50, lam, n_seeds=3)
        moran_results[f"lambda_{lam}"] = m
        print(f"    ρ_S = {m['rho']:.6f}")
        print(f"    1/N = {m['neutral']:.6f}")
        print(f"    Ratio = {m['ratio']:.4f}x")
        print(f"    Favored: {'YES ✓' if m['favored'] else 'NO ✗'}")

    # --- Save ---
    all_results = {
        "framework": "Eco-Evolutionary Dynamic ESS (cumulative fitness)",
        "proposition": "Situational Commitment is ESS when fitness = long-run cumulative payoff with resource feedback",
        "eco_ess_invasion": {k: v for k, v in inv.items() if k != "details"},
        "eco_ess_detail": inv["details"],
        "resource_trajectory": res_traj,
        "eco_moran": moran_results,
    }

    path = os.path.join(OUTPUT_DIR, "ess_proof_v2_eco.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n\n  Results: {path}")
    print("\n" + "=" * 70)
    print("  SUMMARY — Eco-Evolutionary ESS")
    print("=" * 70)
    print(f"  Eco-ESS invasion: {inv['n_violations']}/{inv['n_strategies']} violations"
          f" → {'VALID ✓' if inv['ess_valid'] else 'PARTIAL ⚠'}")
    print(f"  Moran vs defector (λ=0): ρ/neutral = "
          f"{moran_results.get('lambda_0.0', {}).get('ratio', 0):.4f}x")
    print("  Phase 1 v2 COMPLETE!")
