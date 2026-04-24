"""
N2: Partial Observability 확장 실험
EthicaAI Phase N — 관찰 범위 제한 하에서 Meta-Ranking 효과 분석

출력: Fig 33 (관찰 범위 vs 협력률), Fig 34 (Partial Obs ATE 비교)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import os
import sys

# 출력 디렉토리 (환경변수 우선)
OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "ETHICAAI_OUTPUT_DIR", "simulation/outputs/reproduce"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 관찰 범위 설정 (하드코딩 금지)
OBSERVE_RADII = [1, 2, 3, 5, 10, 999]  # 999 = Full Obs
RADIUS_LABELS = ["r=1", "r=2", "r=3", "r=5", "r=10", "Full"]
SVO_CONDITIONS = {"selfish": 0.0, "prosocial": 45.0, "altruistic": 90.0}
N_SEEDS = 10
N_AGENTS = 20
N_ROUNDS = 200


def simulate_partial_obs(observe_radius, svo_theta_deg, seed, use_meta=True):
    """관찰 범위 제한 환경에서 PGG 시뮬레이션"""
    rng = np.random.RandomState(seed)
    svo_theta = np.radians(svo_theta_deg)
    lambda_base = np.sin(svo_theta)

    # 관찰 가능 에이전트 수 (반경에 따라)
    n_observable = min(N_AGENTS, observe_radius * 2 + 1)
    info_ratio = n_observable / N_AGENTS  # 정보 비율

    cooperation_history = np.zeros(N_ROUNDS)
    wealth = np.full(N_AGENTS, 10.0)

    for t in range(N_ROUNDS):
        # 각 에이전트의 λ 결정
        lambdas = np.zeros(N_AGENTS)
        for i in range(N_AGENTS):
            if use_meta:
                # 제한된 관찰: 주변 에이전트만 관찰 가능
                visible_idx = np.arange(max(0, i - observe_radius),
                                        min(N_AGENTS, i + observe_radius + 1))
                visible_wealth = wealth[visible_idx]
                perceived_avg = np.mean(visible_wealth)

                # 동적 λ (불완전 정보 기반)
                uncertainty_penalty = 1.0 - info_ratio * 0.3  # 정보 부족 페널티
                if perceived_avg < 5.0:
                    lambdas[i] = 0.0
                elif perceived_avg > 15.0:
                    lambdas[i] = min(1.0, lambda_base * 1.5 * uncertainty_penalty)
                else:
                    lambdas[i] = lambda_base * uncertainty_penalty
            else:
                lambdas[i] = lambda_base

        # 기여 결정
        contributions = np.zeros(N_AGENTS)
        for i in range(N_AGENTS):
            base_contrib = 0.3 + lambdas[i] * 0.5
            noise = rng.normal(0, 0.1 * (1 - info_ratio * 0.5))
            contributions[i] = np.clip(base_contrib + noise, 0, 1)

        # 공공재 계산
        total_contrib = np.sum(contributions * wealth * 0.3)
        public_return = total_contrib * 1.6 / N_AGENTS

        # 보상 분배
        for i in range(N_AGENTS):
            cost = contributions[i] * wealth[i] * 0.3
            wealth[i] = wealth[i] - cost + public_return + rng.normal(0, 0.2)
            wealth[i] = max(0.1, wealth[i])

        cooperation_history[t] = np.mean(contributions)

    return {
        "final_coop": float(np.mean(cooperation_history[-30:])),
        "final_wealth": float(np.mean(wealth)),
        "gini": float(np.clip(np.std(wealth) / (np.mean(wealth) + 1e-8), 0, 1)),
        "cooperation_history": cooperation_history,
    }


def run_partial_obs_experiment():
    """전체 Partial Obs 실험 실행"""
    results = {}
    for radius, label in zip(OBSERVE_RADII, RADIUS_LABELS):
        results[label] = {}
        for svo_name, svo_theta in SVO_CONDITIONS.items():
            meta_runs = [simulate_partial_obs(radius, svo_theta, s, True) for s in range(N_SEEDS)]
            base_runs = [simulate_partial_obs(radius, svo_theta, s, False) for s in range(N_SEEDS)]
            ate = np.mean([r["final_coop"] for r in meta_runs]) - np.mean([r["final_coop"] for r in base_runs])
            results[label][svo_name] = {
                "meta_coop": float(np.mean([r["final_coop"] for r in meta_runs])),
                "base_coop": float(np.mean([r["final_coop"] for r in base_runs])),
                "ate": float(ate),
                "meta_wealth": float(np.mean([r["final_wealth"] for r in meta_runs])),
            }
    return results


def plot_fig33(results):
    """Fig 33: 관찰 범위 vs 협력률"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'selfish': '#e53935', 'prosocial': '#1e88e5', 'altruistic': '#43a047'}
    markers = {'selfish': 'o', 'prosocial': 's', 'altruistic': 'D'}

    for svo_name in SVO_CONDITIONS:
        meta_vals = [results[r][svo_name]["meta_coop"] for r in RADIUS_LABELS]
        base_vals = [results[r][svo_name]["base_coop"] for r in RADIUS_LABELS]
        x = np.arange(len(RADIUS_LABELS))

        ax.plot(x, meta_vals, color=colors[svo_name], marker=markers[svo_name],
                linewidth=2.5, markersize=8, label=f'{svo_name} (Meta)', zorder=3)
        ax.plot(x, base_vals, color=colors[svo_name], marker=markers[svo_name],
                linewidth=1.5, linestyle='--', markersize=6, label=f'{svo_name} (Base)', alpha=0.5)

    ax.set_xticks(np.arange(len(RADIUS_LABELS)))
    ax.set_xticklabels(RADIUS_LABELS)
    ax.set_xlabel('Observation Radius', fontsize=12)
    ax.set_ylabel('Cooperation Rate', fontsize=12)
    ax.set_title('Fig 33: Meta-Ranking Under Partial Observability', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    path = os.path.join(OUTPUT_DIR, "fig33_partial_obs.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[N2] Fig 33 저장: {path}")


def plot_fig34(results):
    """Fig 34: Partial Obs ATE 히트맵"""
    ate_matrix = np.zeros((len(SVO_CONDITIONS), len(RADIUS_LABELS)))
    svo_names = list(SVO_CONDITIONS.keys())

    for i, svo_name in enumerate(svo_names):
        for j, radius_label in enumerate(RADIUS_LABELS):
            ate_matrix[i, j] = results[radius_label][svo_name]["ate"]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(ate_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.3)

    ax.set_xticks(np.arange(len(RADIUS_LABELS)))
    ax.set_yticks(np.arange(len(svo_names)))
    ax.set_xticklabels(RADIUS_LABELS)
    ax.set_yticklabels([s.capitalize() for s in svo_names])

    for i in range(len(svo_names)):
        for j in range(len(RADIUS_LABELS)):
            ax.text(j, i, f'{ate_matrix[i, j]:+.3f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='black' if abs(ate_matrix[i, j]) < 0.15 else 'white')

    ax.set_title('Fig 34: ATE of Meta-Ranking by Observation Radius', fontsize=14, fontweight='bold')
    ax.set_xlabel('Observation Radius')
    ax.set_ylabel('SVO Condition')
    plt.colorbar(im, label='ATE (Cooperation)')

    path = os.path.join(OUTPUT_DIR, "fig34_partial_obs_ate.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[N2] Fig 34 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [N2] Partial Observability 확장 실험")
    print("=" * 60)

    results = run_partial_obs_experiment()

    print("\n--- Partial Obs ATE Summary ---")
    for radius_label in RADIUS_LABELS:
        for svo_name in SVO_CONDITIONS:
            d = results[radius_label][svo_name]
            print(f"  {radius_label:>6s} | {svo_name:>10s} | Meta={d['meta_coop']:.3f} | Base={d['base_coop']:.3f} | ATE={d['ate']:+.3f}")

    plot_fig33(results)
    plot_fig34(results)

    json_path = os.path.join(OUTPUT_DIR, "partial_obs_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[N2] 결과 JSON: {json_path}")
