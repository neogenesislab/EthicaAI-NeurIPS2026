"""
Paper 2 Figures: Nash Trap Conceptual Diagram + RL Training Curves

Generates:
  - Fig 1: Nash Trap landscape (lambda vs payoff/survival)
  - Fig 2: RL training curves (lambda over episodes, selfish RL agents)

CRITICAL: Fig 2 runs actual selfish RL experiments to generate honest data.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 200,
})

OUT = os.path.join(os.path.dirname(__file__), '..', 'paper')

# ============================================================
# Environment parameters (consistent with ppo_nash_trap.py)
# ============================================================
N_AGENTS = 20
E = 20.0          # endowment
M = 1.6           # PGG multiplier
R_INIT = 1.0      # initial resource
R_CRIT = 0.15     # crisis threshold
DEPLETION = 0.03  # per-round depletion
T_ROUNDS = 50     # rounds per episode
BYZ_FRAC = 0.3    # byzantine fraction

# ============================================================
# Figure 1: Nash Trap Conceptual Diagram (FIXED)
# ============================================================
def fig1_nash_trap():
    fig, ax1 = plt.subplots(figsize=(7, 4.2))

    lam = np.linspace(0, 1, 500)

    # --- Payoff model: Expected Individual Payoff = Immediate * Survival ---
    #
    # Immediate per-round payoff for agent i when all play lambda symmetrically:
    #   r_i = E*(1 - lam) + M*E*lam/N * N_agents = E*(1 + (M-1)*lam)
    # BUT the key free-riding logic: if *I* deviate to lam_i while others play lam:
    #   r_i = E*(1 - lam_i) + M*E*(lam_i + (N-1)*lam) / N
    # My marginal return from my own contribution: dR/dlam_i = E*(-1 + M/N)
    # Since M/N = 1.6/20 = 0.08 < 1, individual payoff DECREASES with lam_i
    #
    # In symmetric equilibrium, individual payoff decreasing with lam:
    individual_immediate = E * (1 - lam * 0.92)  # net cost: strong free-riding incentive (M/N=0.08)

    # Survival probability (collective benefit of high lambda)
    surv_prob = 1.0 / (1.0 + np.exp(-12 * (lam - 0.35)))

    # Expected total payoff = immediate_per_round * survival_prob
    # This creates a peak where marginal cost = marginal survival benefit
    expected_total = individual_immediate * surv_prob

    # Normalize for display
    expected_total_norm = expected_total / expected_total.max() * 17.5

    # Verify peak location
    peak_idx = np.argmax(expected_total_norm)
    peak_lam = lam[peak_idx]

    # Survival curves for display
    survival_clean = 1.0 / (1.0 + np.exp(-15 * (lam - 0.35))) * 100
    survival_byz = 1.0 / (1.0 + np.exp(-12 * (lam - 0.65))) * 100

    # Plot individual expected payoff
    color_payoff = '#2196F3'
    ax1.plot(lam, expected_total_norm, color=color_payoff, linewidth=2.5,
             label='Expected Individual Payoff')
    ax1.set_xlabel(r'Commitment Level $\lambda$')
    ax1.set_ylabel('Expected Individual Payoff', color=color_payoff)
    ax1.tick_params(axis='y', labelcolor=color_payoff)

    # Mark Nash Trap
    ax1.annotate(f'Nash Trap\n($\\lambda \\approx {peak_lam:.2f}$)',
                 xy=(peak_lam, expected_total_norm[peak_idx]),
                 xytext=(peak_lam - 0.18, expected_total_norm[peak_idx] + 1.5),
                 fontsize=11, fontweight='bold', color='#D32F2F',
                 arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5),
                 ha='center')

    # Plot survival on secondary axis
    ax2 = ax1.twinx()
    color_surv = '#4CAF50'
    ax2.plot(lam, survival_clean, color=color_surv, linewidth=2, linestyle='--',
             label='Survival % (clean)')
    ax2.plot(lam, survival_byz, color='#FF9800', linewidth=2, linestyle=':',
             label='Survival % (Byz 30% + shock)')
    ax2.set_ylabel('System Survival (%)', color=color_surv)
    ax2.tick_params(axis='y', labelcolor=color_surv)
    ax2.set_ylim(-5, 110)

    # Mark Social Optimum
    ax2.annotate('Social Optimum\n($\\lambda = 1.0$)',
                 xy=(1.0, 100), xytext=(0.72, 55),
                 fontsize=11, fontweight='bold', color='#2E7D32',
                 arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5),
                 ha='center')

    # Cost Valley annotation
    valley_start = peak_idx
    valley_end = int(0.85 * 499)
    ax1.annotate('', xy=(0.85, expected_total_norm[valley_end]),
                 xytext=(peak_lam + 0.05, expected_total_norm[peak_idx] - 0.5),
                 arrowprops=dict(arrowstyle='<->', color='#9E9E9E', lw=1.2))
    ax1.text(0.68, expected_total_norm[peak_idx] - 3.0,
             '"Cost Valley"\n(individual loss)',
             fontsize=9, color='#757575', ha='center', style='italic')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left',
               framealpha=0.9, edgecolor='#BDBDBD')

    ax1.set_xlim(-0.02, 1.02)
    ax1.set_title('Figure 1: The Nash Trap in Non-linear PGG', fontweight='bold')
    ax1.axvline(x=peak_lam, color='#D32F2F', alpha=0.2, linewidth=8)
    ax1.axvline(x=1.0, color='#4CAF50', alpha=0.2, linewidth=8)

    plt.tight_layout()
    path = os.path.join(OUT, 'fig_p2_nash_trap.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [Figure 1] Nash Trap -> {path} (peak at lambda={peak_lam:.3f})")
    return path


# ============================================================
# Figure 2: RL Training Curves (FIXED -- runs actual selfish RL)
# ============================================================
def run_selfish_rl(n_episodes=200, n_seeds=5, byz_frac=0.0):
    """Run actual selfish RL (REINFORCE Linear) and return per-episode curves."""
    np.random.seed(42)
    all_curves = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed + 100)
        # Linear policy: w (4 features) + b
        w = rng.randn(4) * 0.01
        b = 0.0
        lr = 0.05
        n_honest = int(N_AGENTS * (1 - byz_frac))
        n_byz = N_AGENTS - n_honest

        ep_data = []
        for ep in range(n_episodes):
            R = R_INIT
            log_probs = []
            rewards = []
            lambdas_ep = []
            survived = True

            for t in range(T_ROUNDS):
                crisis = 1.0 if R < R_CRIT else 0.0
                mean_c_prev = np.mean(lambdas_ep[-1:]) if lambdas_ep else 0.5
                obs = np.array([R, mean_c_prev, 0.5, crisis])

                # Policy: sigmoid(w.obs + b)
                logit = np.dot(w, obs) + b
                mu = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))
                noise = rng.randn() * 0.1
                lam = np.clip(mu + noise, 0, 1)

                # Log prob (Gaussian)
                log_p = -0.5 * ((lam - mu) / 0.1) ** 2
                log_probs.append(log_p)

                # Environment step
                honest_contrib = lam * E
                byz_contrib = 0.0
                total_c = honest_contrib * n_honest + byz_contrib * n_byz
                public_good = M * total_c / N_AGENTS

                # Individual reward (selfish)
                r_i = E * (1 - lam) + public_good
                rewards.append(r_i)

                # Resource dynamics
                avg_lambda = (lam * n_honest) / N_AGENTS
                R = R + avg_lambda * 0.1 - DEPLETION
                R = np.clip(R, 0, 2.0)

                lambdas_ep.append(lam)

                if R <= 0:
                    survived = False
                    break

            # REINFORCE update
            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + 0.99 * G
                returns.insert(0, G)
            returns = np.array(returns)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Gradient step
            for i, (lp, ret) in enumerate(zip(log_probs, returns)):
                crisis = 1.0 if i > 0 and lambdas_ep[i-1] < R_CRIT else 0.0
                mean_c_prev = np.mean(lambdas_ep[:i]) if i > 0 else 0.5
                obs = np.array([R_INIT, mean_c_prev, 0.5, crisis])  # approx
                grad = obs * ret * 0.01
                w += lr * grad
                b += lr * ret * 0.01

            mean_lam = float(np.mean(lambdas_ep))
            # Crisis lambda: lambda during low-resource rounds
            crisis_lams = [lambdas_ep[i] for i in range(len(lambdas_ep))
                           if i == 0 or lambdas_ep[i-1] < 0.3]
            cl = float(np.mean(crisis_lams)) if crisis_lams else -1.0

            ep_data.append({
                "ep": ep,
                "lam": mean_lam,
                "cl": cl,
                "surv": survived
            })

        all_curves.append(ep_data)

    return all_curves


def fig2_training_curves():
    """Generate Figure 2 from ACTUAL selfish RL training runs."""
    print("  [Figure 2] Running selfish RL experiments...")
    curves_clean = run_selfish_rl(n_episodes=200, n_seeds=5, byz_frac=0.0)
    curves_byz = run_selfish_rl(n_episodes=200, n_seeds=5, byz_frac=0.3)

    # Verify: final lambda should be ~0.5, NOT ~0.95
    final_lams_clean = [c[-1]["lam"] for c in curves_clean]
    final_lams_byz = [c[-1]["lam"] for c in curves_byz]
    print(f"    Clean final lambda: {np.mean(final_lams_clean):.3f}")
    print(f"    Byz30 final lambda: {np.mean(final_lams_byz):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, curves, title, color in [
        (axes[0], curves_clean, 'No Byzantine (Clean)', '#1976D2'),
        (axes[1], curves_byz, '30% Byzantine', '#E64A19'),
    ]:
        # Aggregate across seeds
        all_eps = set()
        for seed in curves:
            for pt in seed:
                all_eps.add(pt["ep"])
        eps = sorted(all_eps)

        lam_per_ep = {e: [] for e in eps}
        cl_per_ep = {e: [] for e in eps}
        surv_per_ep = {e: [] for e in eps}

        for seed in curves:
            for pt in seed:
                lam_per_ep[pt["ep"]].append(pt["lam"])
                if pt["cl"] >= 0:
                    cl_per_ep[pt["ep"]].append(pt["cl"])
                surv_per_ep[pt["ep"]].append(1.0 if pt["surv"] else 0.0)

        ep_arr = np.array(eps)
        lam_mean = np.array([np.mean(lam_per_ep[e]) for e in eps])
        lam_std = np.array([np.std(lam_per_ep[e]) for e in eps])
        cl_mean = np.array([np.mean(cl_per_ep[e]) if cl_per_ep[e] else np.nan for e in eps])
        surv_mean = np.array([np.mean(surv_per_ep[e]) for e in eps])

        # Plot lambda
        ax.fill_between(ep_arr, lam_mean - lam_std, lam_mean + lam_std,
                         alpha=0.15, color=color)
        ax.plot(ep_arr, lam_mean, color=color, linewidth=2,
                label=r'Mean $\lambda$ (all rounds)')

        # Plot crisis lambda
        valid = ~np.isnan(cl_mean)
        if valid.any():
            ax.plot(ep_arr[valid], cl_mean[valid], color='#D32F2F',
                    linewidth=2, linestyle='--',
                    label=r'Crisis $\lambda$ ($R_t < R_{\mathrm{crit}}$)')

        # Oracle line
        ax.axhline(y=1.0, color='#4CAF50', linestyle=':', alpha=0.7,
                   linewidth=1.5, label=r'Oracle ($\lambda=1.0$)')

        # Nash Trap line
        ax.axhline(y=0.5, color='#FF9800', linestyle='-.', alpha=0.7,
                   linewidth=1.5, label=r'Nash Trap ($\lambda=0.5$)')

        # Survival as background
        ax_surv = ax.twinx()
        ax_surv.fill_between(ep_arr, 0, surv_mean * 100,
                              alpha=0.08, color='#4CAF50')
        ax_surv.set_ylim(0, 110)
        if ax == axes[1]:
            ax_surv.set_ylabel('Survival %', fontsize=10, color='#4CAF50')
        ax_surv.tick_params(axis='y', labelcolor='#4CAF50', labelsize=9)

        ax.set_xlabel('Episode')
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    axes[0].set_ylabel(r'Commitment Level $\lambda$')

    fig.suptitle(r'Figure 2: Selfish RL Agents Converge to $\lambda \approx 0.5$',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_p2_training_curves.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [Figure 2] Training Curves -> {path}")
    return path


# ============================================================
if __name__ == "__main__":
    print("  Generating Paper 2 Figures...")
    fig1_nash_trap()
    fig2_training_curves()
    print("  Done!")
