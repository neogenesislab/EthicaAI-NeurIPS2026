"""
EthicaAI Phase H: Evolutionary Competition Simulation
NeurIPS 2026 — 후속 연구 H1

논문 Section 6.3에서 제안한 '진화적 경쟁 시뮬레이션':
Self-Interest vs Committed(Meta-Ranking) 에이전트 간 장기 진화적 역학

방법:
  1. Mixed Population: N명 중 일부는 Meta-Ranking ON, 나머지는 OFF
  2. 세대(Generation)마다 성과 기반으로 전략 비율이 변화 (Replicator Dynamics)
  3. 어떤 전략이 ESS(진화적으로 안정적인 전략)인지 검증
  4. 이론적 예측(Situational Commitment = ESS)과 시뮬레이션 결과 비교
"""
import os
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- NeurIPS 스타일 설정 ---
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# --- 설정 ---
EVO_CONFIG = {
    "POP_SIZE": 100,        # 전체 인구
    "NUM_GENERATIONS": 200, # 세대 수
    "NUM_INTERACTIONS": 50, # 세대당 상호작용 횟수
    "MUTATION_RATE": 0.01,  # 돌연변이율
    "NUM_SEEDS": 10,        # 반복 횟수
}

# PGG 매개변수
PGG_PARAMS = {
    "ENDOWMENT": 20.0,
    "MULTIPLIER": 1.6,
    "GROUP_SIZE": 4,
}

SVO_THETA_PROSOCIAL = math.pi / 4  # 45도 (Prosocial)


def compute_lambda(wealth, use_meta, survival=-5.0, boost=5.0):
    """Dynamic lambda 계산."""
    lam_base = math.sin(SVO_THETA_PROSOCIAL)
    if not use_meta:
        return lam_base
    if wealth < survival:
        return 0.0
    elif wealth > boost:
        return min(1.0, 1.5 * lam_base)
    else:
        return lam_base


def play_pgg_round(strategies, wealth, rng):
    """
    한 라운드의 PGG.
    strategies: boolean array (True=Meta-Ranking, False=Static)
    """
    n = len(strategies)
    endowment = PGG_PARAMS["ENDOWMENT"]
    multiplier = PGG_PARAMS["MULTIPLIER"]
    group_size = PGG_PARAMS["GROUP_SIZE"]

    # 랜덤 그룹 편성
    indices = rng.permutation(n)
    payoffs = np.zeros(n)

    for g_start in range(0, n - group_size + 1, group_size):
        group = indices[g_start:g_start + group_size]
        contributions = []
        for idx in group:
            lam = compute_lambda(wealth[idx], strategies[idx])
            noise = rng.normal(0, endowment * 0.03)
            c = np.clip(lam * endowment + noise, 0, endowment)
            contributions.append(c)

        contributions = np.array(contributions)
        public_good = np.sum(contributions) * multiplier / group_size

        for j, idx in enumerate(group):
            payoffs[idx] = (endowment - contributions[j]) + public_good

    return payoffs


def replicator_dynamics(strategies, fitness, mutation_rate, rng):
    """진화적 복제 역학."""
    n = len(strategies)
    new_strategies = strategies.copy()

    # 적합도 기반 복제
    meta_fitness = np.mean(fitness[strategies])
    static_fitness = np.mean(fitness[~strategies]) if np.any(~strategies) else 0

    total_fitness = meta_fitness + static_fitness
    if total_fitness > 0:
        meta_frac_target = meta_fitness / total_fitness
    else:
        meta_frac_target = 0.5

    # 일부 에이전트 전략 전환 (점진적)
    for i in range(n):
        if rng.random() < mutation_rate:
            # 무작위 돌연변이
            new_strategies[i] = rng.random() < 0.5
        else:
            # 적합도 비례 전환
            if rng.random() < 0.1:  # 10% 확률로 재평가
                if strategies[i]:
                    # Meta-Ranking 유지할지 평가
                    if fitness[i] < static_fitness and rng.random() < 0.3:
                        new_strategies[i] = False
                else:
                    # Static에서 Meta-Ranking으로 전환 평가
                    if fitness[i] < meta_fitness and rng.random() < 0.3:
                        new_strategies[i] = True

    return new_strategies


def run_evolutionary_simulation(initial_meta_fraction=0.5, seed=0):
    """하나의 진화 시뮬레이션 실행."""
    config = EVO_CONFIG
    rng = np.random.RandomState(seed)
    n = config["POP_SIZE"]

    # 초기 전략 배분
    strategies = np.zeros(n, dtype=bool)
    strategies[:int(n * initial_meta_fraction)] = True
    rng.shuffle(strategies)

    wealth = np.zeros(n)
    history = {
        "meta_fraction": [],
        "avg_meta_fitness": [],
        "avg_static_fitness": [],
        "population_welfare": [],
        "gini": [],
    }

    for gen in range(config["NUM_GENERATIONS"]):
        # 이번 세대의 누적 적합도
        gen_payoffs = np.zeros(n)

        for _ in range(config["NUM_INTERACTIONS"]):
            payoffs = play_pgg_round(strategies, wealth, rng)
            gen_payoffs += payoffs
            wealth += payoffs - PGG_PARAMS["ENDOWMENT"]

        # 기록
        meta_mask = strategies
        meta_frac = float(np.mean(meta_mask))
        avg_meta_fit = float(np.mean(gen_payoffs[meta_mask])) if np.any(meta_mask) else 0
        avg_static_fit = float(np.mean(gen_payoffs[~meta_mask])) if np.any(~meta_mask) else 0

        # Gini
        sorted_payoffs = np.sort(gen_payoffs)
        n_agents = len(sorted_payoffs)
        index = np.arange(1, n_agents + 1)
        gini = float(np.sum((2 * index - n_agents - 1) * sorted_payoffs) /
                     (n_agents * np.sum(sorted_payoffs))) if np.sum(sorted_payoffs) > 0 else 0

        history["meta_fraction"].append(meta_frac)
        history["avg_meta_fitness"].append(avg_meta_fit)
        history["avg_static_fitness"].append(avg_static_fit)
        history["population_welfare"].append(float(np.mean(gen_payoffs)))
        history["gini"].append(gini)

        # 세대 교체 (Replicator Dynamics)
        strategies = replicator_dynamics(strategies, gen_payoffs,
                                         config["MUTATION_RATE"], rng)

    return history


def run_invasion_analysis(output_dir):
    """다양한 초기 비율에서의 진화 역학 분석."""
    initial_fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_results = {}

    print("=" * 65)
    print("  H1: Evolutionary Competition Simulation")
    print(f"  Pop={EVO_CONFIG['POP_SIZE']}, Gen={EVO_CONFIG['NUM_GENERATIONS']}")
    print("=" * 65)

    for init_frac in initial_fractions:
        print(f"\n--- Initial Meta-Ranking Fraction: {init_frac:.0%} ---")
        seed_histories = []
        for seed in range(EVO_CONFIG["NUM_SEEDS"]):
            h = run_evolutionary_simulation(init_frac, seed)
            seed_histories.append(h)

        # 집계
        final_fracs = [h["meta_fraction"][-1] for h in seed_histories]
        avg_final = float(np.mean(final_fracs))
        print(f"  Final Meta-Ranking Fraction: {avg_final:.3f} "
              f"(range: {min(final_fracs):.3f} - {max(final_fracs):.3f})")

        all_results[f"init_{init_frac:.1f}"] = {
            "histories": seed_histories,
            "final_meta_fraction_mean": avg_final,
            "final_meta_fraction_std": float(np.std(final_fracs)),
        }

    return all_results


def plot_evolution(all_results, output_dir):
    """진화적 경쟁 결과 시각화."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    init_fracs = sorted(all_results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(init_fracs)))

    # --- (a) Meta-Ranking Fraction Over Generations ---
    ax = axes[0, 0]
    for idx, key in enumerate(init_fracs):
        histories = all_results[key]["histories"]
        fracs = np.array([h["meta_fraction"] for h in histories])
        mean_frac = np.mean(fracs, axis=0)
        std_frac = np.std(fracs, axis=0)
        gens = np.arange(len(mean_frac))

        label = key.replace("init_", "p0=")
        ax.plot(gens, mean_frac, color=colors[idx], linewidth=2, label=label)
        ax.fill_between(gens, mean_frac - std_frac, mean_frac + std_frac,
                       color=colors[idx], alpha=0.1)

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Meta-Ranking Fraction")
    ax.set_title("(a) Strategy Evolution\nConvergence to ESS")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # --- (b) Fitness Comparison ---
    ax = axes[0, 1]
    # 중간 초기값(0.5)의 결과 사용
    mid_key = [k for k in init_fracs if "0.5" in k][0]
    histories = all_results[mid_key]["histories"]
    meta_fits = np.mean([h["avg_meta_fitness"] for h in histories], axis=0)
    static_fits = np.mean([h["avg_static_fitness"] for h in histories], axis=0)
    gens = np.arange(len(meta_fits))

    ax.plot(gens, meta_fits, color='#2196F3', linewidth=2, label='Meta-Ranking')
    ax.plot(gens, static_fits, color='#F44336', linewidth=2, label='Static (Baseline)')
    ax.fill_between(gens, meta_fits, static_fits, alpha=0.1,
                   where=meta_fits > static_fits, color='#2196F3')
    ax.fill_between(gens, meta_fits, static_fits, alpha=0.1,
                   where=meta_fits < static_fits, color='#F44336')

    ax.set_xlabel("Generation")
    ax.set_ylabel("Average Fitness")
    ax.set_title("(b) Fitness Competition\n(Blue > Red = Meta-Ranking dominates)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- (c) Population Welfare ---
    ax = axes[1, 0]
    for idx, key in enumerate(init_fracs):
        histories = all_results[key]["histories"]
        welfare = np.mean([h["population_welfare"] for h in histories], axis=0)
        label = key.replace("init_", "p0=")
        ax.plot(np.arange(len(welfare)), welfare, color=colors[idx],
                linewidth=2, label=label)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Population Welfare")
    ax.set_title("(c) Social Welfare Over Evolution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (d) ESS Analysis (Final State) ---
    ax = axes[1, 1]
    init_vals = [float(k.replace("init_", "")) for k in init_fracs]
    final_means = [all_results[k]["final_meta_fraction_mean"] for k in init_fracs]
    final_stds = [all_results[k]["final_meta_fraction_std"] for k in init_fracs]

    ax.errorbar(init_vals, final_means, yerr=final_stds,
               fmt='o-', color='#2196F3', linewidth=2, markersize=10,
               capsize=5, label='Final Meta-Ranking Fraction')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No change line')

    # 수렴점 표시
    if len(final_means) > 0:
        avg_converge = np.mean(final_means)
        ax.axhline(y=avg_converge, color='green', linestyle=':',
                  alpha=0.5, label=f'ESS = {avg_converge:.2f}')

    ax.set_xlabel("Initial Meta-Ranking Fraction")
    ax.set_ylabel("Final Meta-Ranking Fraction")
    ax.set_title(f"(d) ESS Convergence Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.suptitle("H1: Evolutionary Competition\n"
                 "Does Situational Commitment survive natural selection?",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(output_dir, "fig_evolutionary_competition.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[H1] Figure 저장: {out_path}")
    return out_path


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "simulation/outputs"
    os.makedirs(output_dir, exist_ok=True)

    all_results = run_invasion_analysis(output_dir)
    plot_evolution(all_results, output_dir)

    # JSON 저장 (histories 제외, 요약만)
    summary = {}
    for key, data in all_results.items():
        summary[key] = {
            "final_meta_fraction_mean": data["final_meta_fraction_mean"],
            "final_meta_fraction_std": data["final_meta_fraction_std"],
        }

    out_json = os.path.join(output_dir, "evolutionary_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[H1] 결과 JSON 저장: {out_json}")
