"""
ESS Analytical Proof — Numerical Verification
Phase 1, Step 1-1 ~ 1-3

Proposition 3: In a repeated N-player PGG with resource dynamics,
Situational Commitment (dynamic λ_t conditioned on R_t) is the unique ESS
among the class of stationary strategies {λ = const ∈ [0,1]}.

ESS Condition: For ANY fixed mutant strategy λ' ∈ [0,1], a population of
Situational agents cannot be invaded by a small fraction ε of λ'-mutants.

Mathematical Framework:
  - N agents, fraction x play Situational, fraction (1-x) play fixed λ'
  - Resource dynamics: R_{t+1} = R_t + α·(mean_contrib/ENDOW - threshold)
  - Situational: λ_t = g(θ, R_t)  [dynamic, resource-conditioned]
  - Mutant: λ' = const  [static]
  - Long-run fitness: average payoff over T→∞ steps at steady-state R*

Key Insight: Situational agents implement a "best-response to resource level"
that no fixed strategy can match across ALL resource regimes.

Includes:
  1. Replicator Dynamics (mean-field limit)
  2. Moran Process fixation probability (finite population, N=50)
  3. PGG-specific ESS boundary: MPCR_critical(N, θ)
"""

import numpy as np
import json
import os

# ============================================================
# 1. PGG Environment Parameters
# ============================================================
ENDOWMENT = 100
MPCR = 1.6
ALPHA = 0.02       # Resource adjustment rate
THRESHOLD = 0.3    # Resource equilibrium target
R_CRISIS = 0.2     # Crisis threshold
R_ABUNDANCE = 0.7  # Abundance threshold
SVO_THETA = np.radians(45)  # Prosocial baseline
T_STEPS = 2000     # Long-run simulation length
T_BURN = 500       # Burn-in period


# ============================================================
# 2. Situational Commitment λ_t Function
# ============================================================
def lambda_situational(theta, R_t):
    """Dynamic λ conditioned on resource level R_t.

    - Crisis (R < 0.2): survival mode, λ → 0.3·sin(θ)
    - Abundance (R > 0.7): generous mode, λ → 1.5·sin(θ)
    - Normal: smooth interpolation
    """
    base = np.sin(theta)
    if R_t < R_CRISIS:
        return max(0.0, 0.3 * base)
    elif R_t > R_ABUNDANCE:
        return min(1.0, 1.5 * base)
    else:
        return base * (0.7 + 1.6 * R_t)


# ============================================================
# 3. Long-Run Fitness Computation
# ============================================================
def compute_long_run_fitness(n_agents, x_sit, lambda_mutant, seed=42, mpcr=None):
    """Compute steady-state average payoffs for Situational and Mutant agents.

    Args:
        n_agents: Total population size N
        x_sit: Fraction of Situational agents
        lambda_mutant: Fixed λ value for mutant strategy
        seed: Random seed
        mpcr: Override MPCR value (default: global MPCR)

    Returns:
        (fitness_sit, fitness_mut, R_steady): long-run average payoffs and steady resource
    """
    if mpcr is None:
        mpcr = MPCR
    rng = np.random.RandomState(seed)
    n_sit = int(round(n_agents * x_sit))
    n_mut = n_agents - n_sit

    R_t = 0.5  # Initial resource
    payoffs_sit = []
    payoffs_mut = []
    R_history = []

    for t in range(T_STEPS):
        # Situational agents' contributions
        lam_s = lambda_situational(SVO_THETA, R_t)
        contribs_sit = np.clip(
            lam_s * ENDOWMENT + rng.normal(0, 3, n_sit), 0, ENDOWMENT
        ) if n_sit > 0 else np.array([])

        # Mutant agents' contributions
        contribs_mut = np.clip(
            lambda_mutant * ENDOWMENT + rng.normal(0, 3, n_mut), 0, ENDOWMENT
        ) if n_mut > 0 else np.array([])

        # PGG dynamics
        all_contribs = np.concatenate([contribs_sit, contribs_mut])
        total_c = all_contribs.sum()
        public_good = total_c * mpcr / n_agents

        # Payoffs
        if n_sit > 0:
            payoff_sit = (ENDOWMENT - contribs_sit) + public_good
            payoffs_sit.append(float(payoff_sit.mean()))
        if n_mut > 0:
            payoff_mut = (ENDOWMENT - contribs_mut) + public_good
            payoffs_mut.append(float(payoff_mut.mean()))

        # Resource dynamics
        R_t = np.clip(R_t + ALPHA * (all_contribs.mean() / ENDOWMENT - THRESHOLD), 0, 1)
        if t >= T_BURN:
            R_history.append(R_t)

    # Long-run average (excluding burn-in)
    f_sit = float(np.mean(payoffs_sit[T_BURN:])) if payoffs_sit else 0.0
    f_mut = float(np.mean(payoffs_mut[T_BURN:])) if payoffs_mut else 0.0
    R_ss = float(np.mean(R_history)) if R_history else 0.5

    return f_sit, f_mut, R_ss


# ============================================================
# 4. ESS Invasion Analysis
# ============================================================
def check_ess_invasion(n_agents=50, epsilon=0.02, n_mutant_strategies=51, n_seeds=5):
    """Test ESS condition: Can ANY fixed-λ mutant invade Situational population?

    For each λ' ∈ [0, 1], compute:
      F_Sit(1-ε) vs F_Mut(1-ε)
    ESS iff F_Sit > F_Mut for ALL λ' ≠ λ_Sit

    Returns: dict with invasion analysis results
    """
    x_sit = 1 - epsilon  # Dominant Situational, small mutant fraction
    lambda_range = np.linspace(0, 1, n_mutant_strategies)

    results = []
    for lam_mut in lambda_range:
        fitness_diffs = []
        for seed in range(n_seeds):
            f_sit, f_mut, R_ss = compute_long_run_fitness(
                n_agents, x_sit, lam_mut, seed=seed
            )
            fitness_diffs.append(f_sit - f_mut)

        mean_diff = float(np.mean(fitness_diffs))
        se_diff = float(np.std(fitness_diffs) / np.sqrt(n_seeds))

        results.append({
            "lambda_mutant": float(lam_mut),
            "fitness_diff_mean": mean_diff,  # F_Sit - F_Mut (positive = ESS holds)
            "fitness_diff_se": se_diff,
            "ess_holds": mean_diff > 0,
        })

    n_ess_violations = sum(1 for r in results if not r["ess_holds"])

    return {
        "n_agents": n_agents,
        "epsilon": epsilon,
        "n_strategies_tested": n_mutant_strategies,
        "n_ess_violations": n_ess_violations,
        "ess_is_valid": n_ess_violations == 0,
        "results": results,
    }


# ============================================================
# 5. Moran Process Fixation Probability
# ============================================================
def moran_fixation_probability(n_agents, lambda_mutant, n_seeds=5):
    """Compute fixation probability of a single Situational mutant
    in a population of n_agents-1 fixed-λ agents via numerical estimation.

    ρ_S = 1 / (1 + Σ_{k=1}^{N-1} Π_{j=1}^{k} f_Mut(j/N) / f_Sit(j/N))

    If ρ_S > 1/N, Situational is favored by selection over neutral drift.
    """
    # Compute fitness ratios at each composition
    fitness_ratios = []  # f_Mut / f_Sit at x = j/N

    for j in range(1, n_agents):
        x = j / n_agents  # fraction of Situational agents
        diffs = []
        for seed in range(n_seeds):
            f_sit, f_mut, _ = compute_long_run_fitness(
                n_agents, x, lambda_mutant, seed=seed
            )
            if f_sit > 0:
                diffs.append(f_mut / f_sit)
            else:
                diffs.append(1.0)
        fitness_ratios.append(float(np.mean(diffs)))

    # Compute fixation probability via Moran formula
    # ρ = 1 / (1 + Σ_{k=1}^{N-1} Π_{j=1}^{k} r_j)
    # where r_j = f_Mut(j/N) / f_Sit(j/N)
    partial_products = []
    running_product = 1.0
    for k in range(len(fitness_ratios)):
        running_product *= fitness_ratios[k]
        partial_products.append(running_product)

    rho = 1.0 / (1.0 + sum(partial_products))
    neutral = 1.0 / n_agents

    return {
        "fixation_probability": float(rho),
        "neutral_drift": float(neutral),
        "favored_by_selection": rho > neutral,
        "selection_ratio": float(rho / neutral),
    }


# ============================================================
# 6. Main Execution
# ============================================================
if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'ess_proof')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  Phase 1: ESS Analytical Proof — Numerical Verification")
    print("=" * 70)

    # --- Step 1: Invasion Analysis ---
    print("\n[Step 1] ESS Invasion Analysis (N=50, ε=0.02, 51 mutant strategies)")
    print("  Testing: Can any fixed-λ mutant invade Situational population?")

    invasion = check_ess_invasion(n_agents=50, epsilon=0.02, n_mutant_strategies=51, n_seeds=5)

    print(f"\n  Strategies tested: {invasion['n_strategies_tested']}")
    print(f"  ESS violations: {invasion['n_ess_violations']}")
    print(f"  ESS VALID: {'YES ✓' if invasion['ess_is_valid'] else 'NO ✗'}")

    # Print detailed results
    print(f"\n  {'λ_mut':>8s} {'F_Sit-F_Mut':>12s} {'SE':>8s} {'ESS':>5s}")
    print("  " + "-" * 40)
    for r in invasion["results"][::5]:  # Every 5th strategy
        marker = "✓" if r["ess_holds"] else "✗"
        print(f"  {r['lambda_mutant']:8.2f} {r['fitness_diff_mean']:12.3f} "
              f"{r['fitness_diff_se']:8.3f} {marker:>5s}")

    # --- Step 2: Moran Fixation (selected mutant types) ---
    print("\n\n[Step 2] Moran Process Fixation Probability (N=50)")
    print("  Testing: Is Situational favored by selection over neutral drift?")

    moran_results = {}
    for lam_test in [0.0, 0.3, 0.5, 0.7, 1.0]:
        print(f"\n  λ_mutant = {lam_test:.1f}...")
        moran = moran_fixation_probability(50, lam_test, n_seeds=3)
        moran_results[f"lambda_{lam_test:.1f}"] = moran
        print(f"    ρ_S = {moran['fixation_probability']:.6f}")
        print(f"    1/N = {moran['neutral_drift']:.6f}")
        print(f"    ρ_S/neutral = {moran['selection_ratio']:.3f}x")
        print(f"    Favored: {'YES ✓' if moran['favored_by_selection'] else 'NO ✗'}")

    # --- Step 3: ESS Boundary (MPCR sweep) ---
    print("\n\n[Step 3] ESS Boundary — MPCR Critical Value")
    print("  Sweeping MPCR to find ESS validity boundary...")

    mpcr_results = []
    for mpcr_test in [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0]:
        # Quick test with defector (λ=0) only
        diffs = []
        for seed in range(3):
            f_s, f_d, _ = compute_long_run_fitness(50, 0.98, 0.0, seed=seed, mpcr=mpcr_test)
            diffs.append(f_s - f_d)
        mean_d = float(np.mean(diffs))
        mpcr_results.append({
            "mpcr": mpcr_test,
            "fitness_diff_vs_defector": mean_d,
            "ess_holds": mean_d > 0,
        })
        marker = "✓" if mean_d > 0 else "✗"
        print(f"  MPCR={mpcr_test:.1f} | F_Sit - F_Def = {mean_d:+.3f} | ESS: {marker}")

    # --- Save results ---
    all_results = {
        "invasion_analysis": {
            k: v for k, v in invasion.items() if k != "results"
        },
        "invasion_detail": invasion["results"],
        "moran_fixation": moran_results,
        "mpcr_boundary": mpcr_results,
    }

    json_path = os.path.join(OUTPUT_DIR, "ess_proof_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n\n  Results saved: {json_path}")
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  ESS Invasion (51 strategies): {'VALID ✓' if invasion['ess_is_valid'] else 'VIOLATED ✗'}")
    print(f"  Moran Fixation (N=50, vs defector): "
          f"ρ={moran_results.get('lambda_0.0', {}).get('fixation_probability', 0):.6f} "
          f"({'> 1/N ✓' if moran_results.get('lambda_0.0', {}).get('favored_by_selection', False) else '< 1/N ✗'})")
    print(f"  MPCR boundary: ESS holds for MPCR ≥ "
          f"{next((r['mpcr'] for r in mpcr_results if r['ess_holds']), 'N/A')}")
    print("\n  Phase 1, Step 1-1 to 1-3 COMPLETE!")
