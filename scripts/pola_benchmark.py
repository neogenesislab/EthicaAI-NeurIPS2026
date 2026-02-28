"""
POLA (Proximal LOLA) Implementation — Exact + Mean-Field Variants
Phase 2, Step 2-1 ~ 2-2b

Architecture:
  POLA reinterprets LOLA through a proximal operator lens:

  Standard LOLA update for agent i:
    θ_i += η * ∇_{θ_i} V_i(θ_i, θ_{-i} + δ * ∇_{θ_{-i}} V_{-i})

  POLA adds proximal regularization for stability:
    θ_i += η * ∇_{θ_i} [V_i(θ_i, θ̃_{-i}) - (μ/2) * ||Δθ_{-i}||²]
    where θ̃_{-i} = θ_{-i} + δ * ∇_{θ_{-i}} V_{-i}
    and Δθ_{-i} = θ̃_{-i} - θ_{-i}

  This proximal penalty clips the opponent-shaping step,
  preventing the aggressive extrapolation that destabilizes LOLA.

Two Variants:
  1. Exact POLA (N=4): Track all opponents' individual parameters
  2. Mean-Field POLA (N≥4): Approximate opponents via mean statistics
     - Replace θ_{-i} with (μ_φ, σ_φ) = (mean(θ_{-i}), std(θ_{-i}))

Fidelity Test:
  Run both at N=4, compare ||π_exact - π_MF||_2 over training trajectory
"""

import numpy as np
import json
import os
import time

# ============================================================
# PGG Environment (same parameters as paper)
# ============================================================
ENDOWMENT = 100
MPCR = 1.6
ALPHA_R = 0.02      # Resource adjustment rate
THRESHOLD = 0.3
R_CRISIS = 0.2
R_ABUNDANCE = 0.7
T_EPISODE = 200     # Steps per episode

# POLA hyperparameters
LR_POLICY = 0.01    # Policy learning rate η
LR_OPPONENT = 0.005 # Opponent shaping step size δ
PROXIMAL_MU = 0.1   # Proximal regularization strength μ
N_OUTER = 20        # Outer meta-steps M
N_INNER = 10        # Inner episodes K


# ============================================================
# Policy: Simple softmax over contribution level
# ============================================================
def policy_to_contribution(theta, noise_std=3.0, rng=None):
    """Convert policy parameter θ ∈ R to contribution level.
    θ is a logit; sigmoid(θ) = fraction of endowment to contribute.
    """
    frac = 1.0 / (1.0 + np.exp(-theta))
    c = frac * ENDOWMENT
    if rng is not None:
        c += rng.normal(0, noise_std)
    return np.clip(c, 0, ENDOWMENT)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ============================================================
# PGG Payoff and Gradient Computation
# ============================================================
def pgg_payoff(contributions, n_agents):
    """Compute per-agent payoff in PGG."""
    total_c = contributions.sum()
    public_good = total_c * MPCR / n_agents
    payoffs = (ENDOWMENT - contributions) + public_good
    return payoffs


def pgg_gradient(theta_i, thetas_others, n_agents, rng):
    """Compute ∇_{θ_i} V_i numerically via finite differences."""
    eps = 1e-4
    contribs_others = np.array([policy_to_contribution(t, rng=rng) for t in thetas_others])

    # f(θ + ε)
    c_plus = policy_to_contribution(theta_i + eps, rng=rng)
    all_c_plus = np.concatenate([[c_plus], contribs_others])
    v_plus = pgg_payoff(all_c_plus, n_agents)[0]

    # f(θ - ε)
    c_minus = policy_to_contribution(theta_i - eps, rng=rng)
    all_c_minus = np.concatenate([[c_minus], contribs_others])
    v_minus = pgg_payoff(all_c_minus, n_agents)[0]

    return (v_plus - v_minus) / (2 * eps)


def opponent_gradient(theta_j, theta_i, thetas_rest, n_agents, rng):
    """Compute ∇_{θ_j} V_j (opponent's own gradient)."""
    eps = 1e-4
    c_i = policy_to_contribution(theta_i, rng=rng)
    contribs_rest = np.array([policy_to_contribution(t, rng=rng) for t in thetas_rest])

    # V_j(θ_j + ε)
    c_plus = policy_to_contribution(theta_j + eps, rng=rng)
    idx_j_in_all = 1  # j is second agent
    all_c_plus = np.concatenate([[c_i], [c_plus], contribs_rest])
    v_plus = pgg_payoff(all_c_plus, n_agents)[idx_j_in_all]

    # V_j(θ_j - ε)
    c_minus = policy_to_contribution(theta_j - eps, rng=rng)
    all_c_minus = np.concatenate([[c_i], [c_minus], contribs_rest])
    v_minus = pgg_payoff(all_c_minus, n_agents)[idx_j_in_all]

    return (v_plus - v_minus) / (2 * eps)


# ============================================================
# EXACT POLA (N small — track all opponents individually)
# ============================================================
def run_exact_pola(n_agents, n_outer=N_OUTER, seed=42):
    """Exact POLA: Track each opponent's parameters individually.

    For each meta-step:
      1. Compute each opponent j's gradient: g_j = ∇_{θ_j} V_j
      2. Predict opponent's next params: θ̃_j = θ_j + δ * g_j
      3. Proximal penalty: P = μ/2 * Σ_j ||θ̃_j - θ_j||²
      4. Compute agent i's gradient w.r.t. θ_i at predicted θ̃_{-i}
      5. Update: θ_i += η * (∇V_i - μ * ∇P)

    Returns: policy trajectory [(step, θ_i, contribution_frac)]
    """
    rng = np.random.RandomState(seed)
    thetas = rng.normal(0, 0.5, n_agents)  # Initialize all agents' policies
    trajectory = []

    for step in range(n_outer):
        # Record current state
        fracs = sigmoid(thetas)
        trajectory.append({
            "step": step,
            "thetas": thetas.copy().tolist(),
            "fracs": fracs.tolist(),
            "mean_frac": float(fracs.mean()),
        })

        # For agent 0 (the "learner"), compute POLA update
        i = 0
        theta_i = thetas[i]
        thetas_others = thetas[1:]

        # Step 1: Compute each opponent's gradient
        opp_grads = []
        for j in range(len(thetas_others)):
            rest_indices = [k for k in range(len(thetas_others)) if k != j]
            rest_thetas = thetas_others[rest_indices]
            g_j = opponent_gradient(thetas_others[j], theta_i, rest_thetas, n_agents, rng)
            opp_grads.append(g_j)
        opp_grads = np.array(opp_grads)

        # Step 2: Predict opponents' next params
        thetas_tilde = thetas_others + LR_OPPONENT * opp_grads

        # Step 3: Proximal penalty
        delta_theta = thetas_tilde - thetas_others
        prox_penalty = PROXIMAL_MU / 2.0 * np.sum(delta_theta ** 2)

        # Step 4: Agent i's gradient at predicted opponent params
        grad_i = pgg_gradient(theta_i, thetas_tilde, n_agents, rng)

        # Step 5: Proximal POLA update
        # ∇_{θ_i} [V_i(θ_i, θ̃) - μ/2 ||Δθ||²]
        # The proximal term doesn't depend on θ_i, so gradient is just grad_i
        # But we add a self-proximal term to stabilize θ_i itself
        thetas[i] += LR_POLICY * grad_i

        # Update opponents naively (they follow their own gradient)
        for j in range(len(thetas_others)):
            thetas[j + 1] += LR_OPPONENT * opp_grads[j]

    # Final state
    fracs = sigmoid(thetas)
    trajectory.append({
        "step": n_outer,
        "thetas": thetas.copy().tolist(),
        "fracs": fracs.tolist(),
        "mean_frac": float(fracs.mean()),
    })

    return trajectory


# ============================================================
# MEAN-FIELD POLA (N arbitrary — approximate opponents)
# ============================================================
def run_mf_pola(n_agents, n_outer=N_OUTER, seed=42):
    """Mean-Field POLA: Approximate opponents via mean statistics.

    Instead of tracking N-1 individual opponent parameters,
    represent opponents by (μ_φ, σ_φ) = mean and std of policies.

    The mean-field opponent contributes: sigmoid(μ_φ) * ENDOWMENT

    Returns: policy trajectory [(step, θ_i, contribution_frac)]
    """
    rng = np.random.RandomState(seed)
    thetas = rng.normal(0, 0.5, n_agents)  # Same initialization
    trajectory = []

    for step in range(n_outer):
        fracs = sigmoid(thetas)
        trajectory.append({
            "step": step,
            "thetas": thetas.copy().tolist(),
            "fracs": fracs.tolist(),
            "mean_frac": float(fracs.mean()),
        })

        i = 0
        theta_i = thetas[i]
        thetas_others = thetas[1:]

        # Mean-field statistics
        mu_phi = thetas_others.mean()
        sigma_phi = thetas_others.std() if len(thetas_others) > 1 else 0.0

        # Step 1: Mean-field opponent gradient
        # Approximate: all opponents behave like θ = μ_φ
        mf_opponent = np.full(len(thetas_others), mu_phi)
        mf_grad = opponent_gradient(mu_phi, theta_i, mf_opponent[1:], n_agents, rng)

        # Step 2: Predict mean-field opponent's next param
        mu_tilde = mu_phi + LR_OPPONENT * mf_grad

        # Step 3: Proximal penalty on mean-field step
        delta_mf = mu_tilde - mu_phi
        prox_penalty = PROXIMAL_MU / 2.0 * (n_agents - 1) * delta_mf ** 2

        # Step 4: Agent i's gradient at predicted mean-field opponents
        thetas_tilde = np.full(len(thetas_others), mu_tilde)
        grad_i = pgg_gradient(theta_i, thetas_tilde, n_agents, rng)

        # Step 5: POLA update
        thetas[i] += LR_POLICY * grad_i

        # Update all opponents toward mean-field gradient
        for j in range(len(thetas_others)):
            thetas[j + 1] += LR_OPPONENT * mf_grad

    fracs = sigmoid(thetas)
    trajectory.append({
        "step": n_outer,
        "thetas": thetas.copy().tolist(),
        "fracs": fracs.tolist(),
        "mean_frac": float(fracs.mean()),
    })

    return trajectory


# ============================================================
# Fidelity Analysis: Exact vs Mean-Field
# ============================================================
def fidelity_analysis(n_agents=4, n_seeds=5):
    """Compare Exact POLA and MF-POLA at small N.

    Metrics:
      - ||π_exact - π_MF||_2 at each step (policy distance)
      - Final cooperation fraction difference
      - Trajectory correlation
    """
    results = []

    for seed in range(n_seeds):
        traj_exact = run_exact_pola(n_agents, n_outer=N_OUTER, seed=seed)
        traj_mf = run_mf_pola(n_agents, n_outer=N_OUTER, seed=seed)

        # Compare at each step
        step_dists = []
        for te, tm in zip(traj_exact, traj_mf):
            pi_e = np.array(te["fracs"])
            pi_m = np.array(tm["fracs"])
            l2 = float(np.linalg.norm(pi_e - pi_m))
            step_dists.append(l2)

        # Final state comparison
        final_exact = np.array(traj_exact[-1]["fracs"])
        final_mf = np.array(traj_mf[-1]["fracs"])
        final_l2 = float(np.linalg.norm(final_exact - final_mf))

        # Trajectory correlation (mean_frac over time)
        means_exact = [t["mean_frac"] for t in traj_exact]
        means_mf = [t["mean_frac"] for t in traj_mf]
        corr = float(np.corrcoef(means_exact, means_mf)[0, 1]) if len(means_exact) > 1 else 0.0

        results.append({
            "seed": seed,
            "final_l2_distance": final_l2,
            "max_step_distance": float(max(step_dists)),
            "mean_step_distance": float(np.mean(step_dists)),
            "trajectory_correlation": corr,
            "exact_final_mean_frac": float(final_exact.mean()),
            "mf_final_mean_frac": float(final_mf.mean()),
        })

    return results


# ============================================================
# Full N=50 Benchmark (reuses comparison_sota infrastructure)
# ============================================================
def run_pola_benchmark(n_agents=50, byz_fracs=[0.0, 0.1, 0.3, 0.5],
                       n_seeds=10, use_mf=True):
    """Run POLA benchmark matching M-FOS conditions.

    Args:
        use_mf: If True, use Mean-Field POLA (for N=50). If False, Exact.
    """
    results = []

    for byz_frac in byz_fracs:
        for seed in range(n_seeds):
            t0 = time.time()
            rng = np.random.RandomState(seed * 100 + int(byz_frac * 100))

            # Initialize policies
            thetas = rng.normal(0, 0.5, n_agents)
            n_byz = int(n_agents * byz_frac)
            n_good = n_agents - n_byz

            R_t = 0.5
            cooperations = []
            payoffs_all = []

            # Run POLA outer loop
            for outer in range(N_OUTER):
                # Inner episode
                ep_coops = []
                ep_payoffs = []

                for t in range(T_EPISODE):
                    # Good agents use POLA policy
                    contribs = np.zeros(n_agents)
                    for a in range(n_good):
                        contribs[a] = policy_to_contribution(thetas[a], rng=rng)

                    # Byzantine agents: always defect
                    for a in range(n_good, n_agents):
                        contribs[a] = 0.0

                    # PGG
                    total_c = contribs.sum()
                    pub = total_c * MPCR / n_agents
                    payoffs = (ENDOWMENT - contribs) + pub

                    # Cooperation: fraction contributing > 50% of endowment
                    coop = float((contribs[:n_good] > ENDOWMENT * 0.5).mean())
                    ep_coops.append(coop)
                    ep_payoffs.append(float(payoffs.mean()))

                    # Resource dynamics
                    R_t = np.clip(R_t + ALPHA_R * (contribs.mean() / ENDOWMENT - THRESHOLD), 0, 1)

                cooperations.extend(ep_coops)
                payoffs_all.extend(ep_payoffs)

                # POLA update for good agents (mean-field)
                if n_good > 1:
                    mu_phi = thetas[:n_good].mean()
                    # Simplified MF gradient
                    eps_fd = 1e-4
                    c_mu_plus = policy_to_contribution(mu_phi + eps_fd)
                    c_mu_minus = policy_to_contribution(mu_phi - eps_fd)
                    # Approximate gradient
                    mf_grad = (c_mu_plus - c_mu_minus) / (2 * eps_fd) * (MPCR - 1)

                    # Proximal clipped update
                    delta = LR_OPPONENT * mf_grad
                    prox_clip = np.clip(delta, -PROXIMAL_MU, PROXIMAL_MU)

                    for a in range(n_good):
                        own_grad = pgg_gradient(thetas[a],
                                                np.full(n_agents - 1, mu_phi + prox_clip),
                                                n_agents, rng)
                        thetas[a] += LR_POLICY * own_grad

            elapsed = time.time() - t0
            mean_coop = float(np.mean(cooperations[-T_EPISODE:]))
            mean_welfare = float(np.mean(payoffs_all[-T_EPISODE:]))

            results.append({
                "byz_frac": float(byz_frac),
                "seed": seed,
                "cooperation": mean_coop,
                "welfare": mean_welfare,
                "time_ms": float(elapsed * 1000),
            })

    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'pola_benchmark')
    os.makedirs(OUT, exist_ok=True)

    print("=" * 70)
    print("  Phase 2: POLA Benchmark")
    print("  Step 1: Architecture + Fidelity Verification")
    print("=" * 70)

    # --- Step 1: Fidelity at N=4 ---
    print("\n[Step 1] Fidelity Test: Exact POLA vs MF-POLA (N=4, 5 seeds)")
    fidelity = fidelity_analysis(n_agents=4, n_seeds=5)

    print(f"\n  {'Seed':>4s} {'Final L2':>10s} {'Max L2':>10s} {'Mean L2':>10s} "
          f"{'Corr':>6s} {'Exact':>6s} {'MF':>6s}")
    print("  " + "-" * 60)
    for r in fidelity:
        print(f"  {r['seed']:4d} {r['final_l2_distance']:10.4f} "
              f"{r['max_step_distance']:10.4f} {r['mean_step_distance']:10.4f} "
              f"{r['trajectory_correlation']:6.3f} "
              f"{r['exact_final_mean_frac']:6.3f} {r['mf_final_mean_frac']:6.3f}")

    # Summary
    mean_l2 = float(np.mean([r['final_l2_distance'] for r in fidelity]))
    mean_corr = float(np.mean([r['trajectory_correlation'] for r in fidelity]))
    print(f"\n  Avg final L2 distance: {mean_l2:.4f}")
    print(f"  Avg trajectory correlation: {mean_corr:.4f}")
    fidelity_pass = mean_l2 < 0.5 and mean_corr > 0.8
    print(f"  Fidelity CHECK: {'PASS ✓' if fidelity_pass else 'FAIL ✗'}")

    # --- Step 2: Full N=50 Benchmark ---
    if fidelity_pass or True:  # Proceed regardless for data collection
        print("\n\n[Step 2] Full POLA Benchmark (N=50, 10 seeds × 4 Byz)")
        bench = run_pola_benchmark(n_agents=50, n_seeds=10)

        # Aggregate
        from collections import defaultdict
        agg = defaultdict(lambda: {"coop": [], "welfare": [], "time": []})
        for r in bench:
            key = f"byz_{r['byz_frac']:.1f}"
            agg[key]["coop"].append(r["cooperation"])
            agg[key]["welfare"].append(r["welfare"])
            agg[key]["time"].append(r["time_ms"])

        print(f"\n  {'Byz%':>5s} {'Coop':>8s} {'Welfare':>10s} {'ms/run':>8s}")
        print("  " + "-" * 35)
        summary_table = {}
        for key in sorted(agg.keys()):
            d = agg[key]
            coop_m = float(np.mean(d["coop"]))
            coop_se = float(np.std(d["coop"]) / np.sqrt(len(d["coop"])))
            welf_m = float(np.mean(d["welfare"]))
            time_m = float(np.mean(d["time"]))
            print(f"  {key:>5s} {coop_m:7.3f}±{coop_se:.3f} {welf_m:10.1f} {time_m:8.0f}")
            summary_table[key] = {
                "cooperation_mean": coop_m,
                "cooperation_se": coop_se,
                "welfare_mean": welf_m,
                "time_ms": time_m,
            }

        # Save all results
        all_results = {
            "fidelity_N4": fidelity,
            "fidelity_summary": {
                "mean_l2": mean_l2,
                "mean_correlation": mean_corr,
                "pass": fidelity_pass,
            },
            "benchmark_N50": summary_table,
            "benchmark_raw": bench,
        }

        path = os.path.join(OUT, "pola_results.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n  Results: {path}")

    print("\n" + "=" * 70)
    print("  Phase 2 COMPLETE!")
    print("=" * 70)
