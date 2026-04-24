"""
EthicaAI N-Player Public Goods Game (PGG) Environment
NeurIPS 2026 — Phase G5

경제학 실험에서 가장 널리 사용되는 N-Player PGG를 JAX로 구현합니다.
이를 통해 Human-AI 비교를 '구조적으로 동일한 환경'에서 수행할 수 있습니다.

규칙:
  1. 각 에이전트는 초기 자금(endowment)에서 기여금(contribution)을 결정
  2. 모든 기여금의 합 × multiplier를 N명에게 균등 분배
  3. '무임승차(Free-Riding)'가 내쉬 균형이지만, 전원 기여가 사회 최적

Meta-Ranking 적용:
  - lambda_t에 따라 기여금 결정이 달라짐
  - 자원(wealth)이 낮으면 기여 감소, 높으면 기여 증가
"""
import os
import json
import math
import numpy as np
import scipy.stats as stats
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
PGG_CONFIG = {
    "N_AGENTS": 4,          # 그룹 크기 (인간 실험과 동일)
    "ENDOWMENT": 20.0,      # 초기 자금
    "MULTIPLIER": 1.6,      # 공공재 곱셈 계수 (1 < m < N이면 딜레마 성립)
    "NUM_ROUNDS": 10,       # 반복 횟수 (인간 실험: 보통 10라운드)
    "NUM_SEEDS": 20,        # 시드 수 (통계적 검정력 확보)
}

SVO_THETAS = {
    "selfish": 0.0,
    "individualist": math.pi / 12,
    "competitive": math.pi / 6,
    "prosocial": math.pi / 4,
    "cooperative": math.pi / 3,
    "altruistic": 5 * math.pi / 12,
    "full_altruist": math.pi / 2,
}

# --- 인간 PGG 데이터 (경험적 평균, Chaudhuri 2011; Fehr & Gachter 2000) ---
HUMAN_PGG_DATA = {
    "contribution_rate_round1": 0.54,    # 1라운드 평균 기여율 (~54%)
    "contribution_rate_round10": 0.22,   # 10라운드 평균 기여율 (~22%, 감소 경향)
    "conditional_cooperator_pct": 0.50,  # 조건부 협력자 비율 (~50%)
    "free_rider_pct": 0.30,             # 무임승차자 비율 (~30%)
    "altruist_pct": 0.14,               # 무조건 협력자 비율 (~14%)
    "decay_rate": -0.035,               # 라운드별 기여율 감소율
}


def compute_lambda(svo_theta, wealth, use_meta=True,
                   survival=-5.0, boost=5.0, beta=0.1):
    """Sen's dynamic lambda_t 계산."""
    lam_base = math.sin(svo_theta)
    if not use_meta:
        return lam_base

    if wealth < survival:
        return 0.0
    elif wealth > boost:
        return min(1.0, 1.5 * lam_base)
    else:
        return lam_base


def decide_contribution(wealth, endowment, svo_theta, use_meta, rng,
                        group_avg_prev=None):
    """에이전트의 PGG 기여금 결정."""
    lam = compute_lambda(svo_theta, wealth, use_meta)

    # 기본 전략: lambda에 비례하여 기여
    # lambda=0 → 무기여, lambda=1 → 전액 기여
    base_contribution = lam * endowment

    # 조건부 협력 요소: 이전 라운드 그룹 평균에 반응
    if group_avg_prev is not None and use_meta:
        # 다른 사람이 많이 내면 나도 더 냄 (Conditional Cooperation)
        reciprocity = 0.3 * (group_avg_prev - base_contribution)
        base_contribution += reciprocity

    # 노이즈 추가 (인간적 불확실성)
    noise = rng.normal(0, endowment * 0.05)
    contribution = np.clip(base_contribution + noise, 0, endowment)

    return float(contribution)


def simulate_pgg(svo_theta, seed, use_meta=True, config=None):
    """N-Player PGG 시뮬레이션."""
    if config is None:
        config = PGG_CONFIG

    rng = np.random.RandomState(seed)
    n = config["N_AGENTS"]
    endowment = config["ENDOWMENT"]
    multiplier = config["MULTIPLIER"]
    num_rounds = config["NUM_ROUNDS"]

    # 에이전트 상태
    wealth = np.zeros(n)
    contributions_per_round = []
    payoffs_per_round = []
    group_avg_prev = None

    for t in range(num_rounds):
        # 각 에이전트의 기여금 결정
        contributions = []
        for i in range(n):
            c = decide_contribution(wealth[i], endowment, svo_theta,
                                   use_meta, rng, group_avg_prev)
            contributions.append(c)

        contributions = np.array(contributions)
        total_contrib = np.sum(contributions)

        # 공공재 분배
        public_good = total_contrib * multiplier / n
        # 각자의 보상 = (자금 - 기여금) + 공공재
        payoffs = (endowment - contributions) + public_good

        wealth += payoffs - endowment  # 누적 수익
        contributions_per_round.append(contributions.tolist())
        payoffs_per_round.append(payoffs.tolist())
        group_avg_prev = float(np.mean(contributions))

    # 결과 집계
    contribution_rates = [np.mean(c) / endowment for c in contributions_per_round]

    return {
        "contribution_rate_r1": float(contribution_rates[0]),
        "contribution_rate_r10": float(contribution_rates[-1]),
        "contribution_rates": [float(cr) for cr in contribution_rates],
        "avg_contribution_rate": float(np.mean(contribution_rates)),
        "avg_payoff": float(np.mean(payoffs_per_round)),
        "gini": float(compute_gini(np.array(payoffs_per_round[-1]))),
        "decay_rate": float(np.polyfit(range(num_rounds), contribution_rates, 1)[0]),
    }


def compute_gini(values):
    """Gini coefficient 계산."""
    values = np.sort(values)
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * values) / (n * np.sum(values)))


def run_pgg_experiment(output_dir):
    """전체 PGG 실험 실행."""
    results = {"full_model": {}, "baseline": {}}

    print("=" * 65)
    print("  G5: N-Player Public Goods Game Experiment")
    print(f"  N={PGG_CONFIG['N_AGENTS']}, Rounds={PGG_CONFIG['NUM_ROUNDS']}, "
          f"Multiplier={PGG_CONFIG['MULTIPLIER']}")
    print("=" * 65)

    for model_key, use_meta in [("full_model", True), ("baseline", False)]:
        print(f"\n--- {model_key.upper()} ---")
        for svo_name, theta in SVO_THETAS.items():
            seed_results = []
            for seed in range(PGG_CONFIG["NUM_SEEDS"]):
                r = simulate_pgg(theta, seed, use_meta)
                seed_results.append(r)

            agg = {
                "contrib_r1": float(np.mean([r["contribution_rate_r1"] for r in seed_results])),
                "contrib_r10": float(np.mean([r["contribution_rate_r10"] for r in seed_results])),
                "avg_contrib": float(np.mean([r["avg_contribution_rate"] for r in seed_results])),
                "avg_payoff": float(np.mean([r["avg_payoff"] for r in seed_results])),
                "gini": float(np.mean([r["gini"] for r in seed_results])),
                "decay_rate": float(np.mean([r["decay_rate"] for r in seed_results])),
                "contrib_curves": [np.mean([r["contribution_rates"][t]
                                           for r in seed_results])
                                  for t in range(PGG_CONFIG["NUM_ROUNDS"])],
            }
            results[model_key][svo_name] = agg
            print(f"  {svo_name:18s} | R1: {agg['contrib_r1']:.3f} "
                  f"| R10: {agg['contrib_r10']:.3f} "
                  f"| Decay: {agg['decay_rate']:.4f}")

    return results


def plot_pgg_results(results, output_dir):
    """PGG 결과 시각화 (인간 데이터 비교 포함)."""
    fig = plt.figure(figsize=(18, 12))

    svo_labels = list(SVO_THETAS.keys())
    x = np.arange(len(svo_labels))

    # --- (a) Contribution Rate Over Rounds ---
    ax1 = fig.add_subplot(2, 2, 1)
    rounds = range(1, PGG_CONFIG["NUM_ROUNDS"] + 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(svo_labels)))

    for idx, svo in enumerate(svo_labels):
        curve = results["full_model"][svo]["contrib_curves"]
        ax1.plot(rounds, curve, '-o', color=colors[idx], label=svo,
                linewidth=1.5, markersize=4)

    # 인간 데이터 오버레이
    human_curve = [HUMAN_PGG_DATA["contribution_rate_round1"] +
                   HUMAN_PGG_DATA["decay_rate"] * t for t in range(PGG_CONFIG["NUM_ROUNDS"])]
    ax1.plot(rounds, human_curve, 'k--', linewidth=3, alpha=0.7,
            label='Human (Avg)', zorder=10)
    ax1.fill_between(rounds,
                     [h - 0.1 for h in human_curve],
                     [h + 0.1 for h in human_curve],
                     color='gray', alpha=0.15, label='Human Range')

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Contribution Rate")
    ax1.set_title("(a) Contribution Decay Over Rounds\n(AI vs Human)")
    ax1.legend(ncol=2, fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # --- (b) R1 vs R10 Comparison ---
    ax2 = fig.add_subplot(2, 2, 2)
    width = 0.25
    r1_full = [results["full_model"][s]["contrib_r1"] for s in svo_labels]
    r10_full = [results["full_model"][s]["contrib_r10"] for s in svo_labels]

    ax2.bar(x - width, r1_full, width, label='Round 1', color='#4CAF50', alpha=0.8)
    ax2.bar(x, r10_full, width, label='Round 10', color='#F44336', alpha=0.8)
    ax2.bar(x + width,
           [HUMAN_PGG_DATA["contribution_rate_round1"]] * len(svo_labels),
           width, label='Human R1', color='gray', alpha=0.4, hatch='//')

    ax2.set_xticks(x)
    ax2.set_xticklabels(svo_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel("Contribution Rate")
    ax2.set_title("(b) Round 1 vs Round 10")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- (c) Meta-Ranking Effect ---
    ax3 = fig.add_subplot(2, 2, 3)
    full_contribs = [results["full_model"][s]["avg_contrib"] for s in svo_labels]
    base_contribs = [results["baseline"][s]["avg_contrib"] for s in svo_labels]

    ax3.bar(x - width/2, base_contribs, width, label='Baseline',
           color='#9E9E9E', alpha=0.8)
    ax3.bar(x + width/2, full_contribs, width, label='Meta-Ranking',
           color='#2196F3', alpha=0.8)

    ax3.set_xticks(x)
    ax3.set_xticklabels(svo_labels, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel("Avg Contribution Rate")
    ax3.set_title("(c) Meta-Ranking Effect on PGG")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- (d) Human-AI Alignment (Wasserstein) ---
    ax4 = fig.add_subplot(2, 2, 4)

    # Prosocial 조건의 기여 곡선과 인간 곡선의 WD 계산
    from scipy.stats import wasserstein_distance

    wd_per_svo = {}
    for svo in svo_labels:
        ai_curve = np.array(results["full_model"][svo]["contrib_curves"])
        human_c = np.array(human_curve[:len(ai_curve)])
        wd = wasserstein_distance(ai_curve, human_c)
        wd_per_svo[svo] = wd

    bars = ax4.bar(x, [wd_per_svo[s] for s in svo_labels],
                  color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(svo_labels))))
    ax4.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='WD=0.2 threshold')

    for i, svo in enumerate(svo_labels):
        wd = wd_per_svo[svo]
        ax4.text(i, wd + 0.005, f"{wd:.3f}", ha='center', va='bottom', fontsize=8)

    ax4.set_xticks(x)
    ax4.set_xticklabels(svo_labels, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel("Wasserstein Distance")
    ax4.set_title("(d) Human-AI Alignment per SVO\n(Lower = More Human-like)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 어떤 SVO가 가장 인간과 유사한지 강조
    best_svo = min(wd_per_svo, key=wd_per_svo.get)
    best_idx = svo_labels.index(best_svo)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

    plt.suptitle("G5: N-Player Public Goods Game\n"
                 "Structural Validation Against Human Behavioral Data",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(output_dir, "fig_pgg_experiment.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[G5] Figure 저장: {out_path}")

    # 가장 인간과 유사한 SVO 출력
    print(f"\n[G5] Most human-like SVO: {best_svo} (WD={wd_per_svo[best_svo]:.4f})")
    return out_path, wd_per_svo


if __name__ == "__main__":
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "simulation/outputs"
    os.makedirs(output_dir, exist_ok=True)

    results = run_pgg_experiment(output_dir)
    fig_path, wd_scores = plot_pgg_results(results, output_dir)

    out_json = os.path.join(output_dir, "pgg_results.json")
    # numpy float를 json serializable로 변환
    serializable = {}
    for model, svos in results.items():
        serializable[model] = {}
        for svo, data in svos.items():
            serializable[model][svo] = {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                                        for k, v in data.items()}
    serializable["human_ai_wd"] = {k: float(v) for k, v in wd_scores.items()}

    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"[G5] 결과 JSON 저장: {out_json}")
