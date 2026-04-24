"""
N3: Multi-Resource 환경 실험
EthicaAI Phase N — 2-자원 PGG (식량 + 환경) 트레이드오프 분석

출력: Fig 35 (자원별 기여 분배), Fig 36 (Multi-Resource ATE)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import os
import sys

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "ETHICAAI_OUTPUT_DIR", "simulation/outputs/reproduce"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 환경 설정 (하드코딩 금지)
RESOURCE_CONFIGS = {
    "Food": {"multiplier": 2.0, "decay": 0.05, "label": "식량(Food)"},
    "Environment": {"multiplier": 1.3, "decay": 0.02, "label": "환경(Env)"},
}
SVO_CONDITIONS = {"selfish": 0.0, "individualist": 15.0, "prosocial": 45.0, "altruistic": 90.0}
N_AGENTS = 20
N_ROUNDS = 200
N_SEEDS = 10


def simulate_multi_resource(svo_theta_deg, seed, use_meta=True):
    """2-자원 PGG 시뮬레이션"""
    rng = np.random.RandomState(seed)
    svo_theta = np.radians(svo_theta_deg)
    lambda_base = np.sin(svo_theta)

    endowment = 10.0
    wealth = np.full(N_AGENTS, endowment)
    food_pool = 50.0
    env_pool = 50.0

    # 기록
    food_contribs = np.zeros(N_ROUNDS)
    env_contribs = np.zeros(N_ROUNDS)
    total_welfare = np.zeros(N_ROUNDS)
    lambda_history = np.zeros(N_ROUNDS)

    for t in range(N_ROUNDS):
        # 동적 λ
        if use_meta:
            resource_pressure = (food_pool + env_pool) / 100.0
            if resource_pressure < 0.3:
                lambda_t = 0.0
            elif resource_pressure > 0.8:
                lambda_t = min(1.0, lambda_base * 1.5)
            else:
                lambda_t = lambda_base
        else:
            lambda_t = lambda_base

        lambda_history[t] = lambda_t

        # 에이전트별 기여 결정 (2차원: food, env)
        agent_food_c = np.zeros(N_AGENTS)
        agent_env_c = np.zeros(N_AGENTS)

        for i in range(N_AGENTS):
            total_budget = wealth[i] * 0.3  # 소득의 30% 기여 가능

            # 자원 간 배분 결정 — λ가 높을수록 환경 비중 증가
            env_ratio = 0.3 + lambda_t * 0.4 + rng.normal(0, 0.05)
            env_ratio = np.clip(env_ratio, 0.1, 0.9)

            agent_food_c[i] = total_budget * (1 - env_ratio)
            agent_env_c[i] = total_budget * env_ratio

        # 공공재 계산
        food_return = np.sum(agent_food_c) * RESOURCE_CONFIGS["Food"]["multiplier"] / N_AGENTS
        env_return = np.sum(agent_env_c) * RESOURCE_CONFIGS["Environment"]["multiplier"] / N_AGENTS

        # 자원 풀 업데이트
        food_pool += np.sum(agent_food_c) * 0.5 - food_pool * RESOURCE_CONFIGS["Food"]["decay"]
        env_pool += np.sum(agent_env_c) * 0.5 - env_pool * RESOURCE_CONFIGS["Environment"]["decay"]
        food_pool = max(1.0, food_pool)
        env_pool = max(1.0, env_pool)

        # 보상
        for i in range(N_AGENTS):
            cost = agent_food_c[i] + agent_env_c[i]
            reward = food_return + env_return * 0.8 + rng.normal(0, 0.2)
            wealth[i] = wealth[i] - cost + reward
            wealth[i] = max(0.5, wealth[i])

        food_contribs[t] = np.mean(agent_food_c)
        env_contribs[t] = np.mean(agent_env_c)
        total_welfare[t] = np.sum(wealth)

    return {
        "food_contribs": food_contribs,
        "env_contribs": env_contribs,
        "welfare": total_welfare,
        "lambda_history": lambda_history,
        "final_food_contrib": float(np.mean(food_contribs[-30:])),
        "final_env_contrib": float(np.mean(env_contribs[-30:])),
        "final_welfare": float(np.mean(total_welfare[-30:])),
        "env_ratio": float(np.mean(env_contribs[-30:]) / (np.mean(food_contribs[-30:]) + np.mean(env_contribs[-30:]) + 1e-8)),
    }


def run_experiment():
    results = {}
    for svo_name, svo_theta in SVO_CONDITIONS.items():
        meta_runs = [simulate_multi_resource(svo_theta, s, True) for s in range(N_SEEDS)]
        base_runs = [simulate_multi_resource(svo_theta, s, False) for s in range(N_SEEDS)]
        results[svo_name] = {
            "meta_welfare": float(np.mean([r["final_welfare"] for r in meta_runs])),
            "base_welfare": float(np.mean([r["final_welfare"] for r in base_runs])),
            "meta_env_ratio": float(np.mean([r["env_ratio"] for r in meta_runs])),
            "base_env_ratio": float(np.mean([r["env_ratio"] for r in base_runs])),
            "ate_welfare": float(np.mean([r["final_welfare"] for r in meta_runs]) - np.mean([r["final_welfare"] for r in base_runs])),
            "meta_food_c": float(np.mean([r["final_food_contrib"] for r in meta_runs])),
            "meta_env_c": float(np.mean([r["final_env_contrib"] for r in meta_runs])),
            "base_food_c": float(np.mean([r["final_food_contrib"] for r in base_runs])),
            "base_env_c": float(np.mean([r["final_env_contrib"] for r in base_runs])),
            "meta_runs": meta_runs,
            "base_runs": base_runs,
        }
    return results


def plot_fig35(results):
    """Fig 35: 자원별 기여 분배 (Stacked Bar)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    svo_names = list(SVO_CONDITIONS.keys())
    x = np.arange(len(svo_names))
    width = 0.35

    # Meta ON
    ax = axes[0]
    food_vals = [results[s]["meta_food_c"] for s in svo_names]
    env_vals = [results[s]["meta_env_c"] for s in svo_names]
    ax.bar(x, food_vals, width, label='Food', color='#ff9800', alpha=0.85)
    ax.bar(x, env_vals, width, bottom=food_vals, label='Environment', color='#4caf50', alpha=0.85)
    ax.set_title('Meta-Ranking ON', fontsize=12, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in svo_names], rotation=15)
    ax.set_ylabel('Mean Contribution'); ax.legend(); ax.grid(True, axis='y', alpha=0.3)

    # Base
    ax = axes[1]
    food_vals = [results[s]["base_food_c"] for s in svo_names]
    env_vals = [results[s]["base_env_c"] for s in svo_names]
    ax.bar(x, food_vals, width, label='Food', color='#ff9800', alpha=0.4)
    ax.bar(x, env_vals, width, bottom=food_vals, label='Environment', color='#4caf50', alpha=0.4)
    ax.set_title('Baseline (Meta OFF)', fontsize=12, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in svo_names], rotation=15)
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Fig 35: Multi-Resource Contribution Allocation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig35_multi_resource.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[N3] Fig 35 저장: {path}")


def plot_fig36(results):
    """Fig 36: Multi-Resource ATE + 환경 배분 비율"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    svo_names = list(SVO_CONDITIONS.keys())
    x = np.arange(len(svo_names))

    # ATE
    ates = [results[s]["ate_welfare"] for s in svo_names]
    colors = ['#e53935' if v < 0 else '#43a047' for v in ates]
    ax1.bar(x, ates, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax1.set_title('ATE (Welfare)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels([s.capitalize() for s in svo_names], rotation=15)
    ax1.set_ylabel('ΔWelfare'); ax1.axhline(0, color='black', linewidth=0.5)
    ax1.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(ates):
        ax1.text(i, v + 0.5 * np.sign(v), f'{v:+.1f}', ha='center', fontweight='bold', fontsize=10)

    # 환경 배분 비율
    meta_ratios = [results[s]["meta_env_ratio"] * 100 for s in svo_names]
    base_ratios = [results[s]["base_env_ratio"] * 100 for s in svo_names]
    ax2.bar(x - 0.15, meta_ratios, 0.3, label='Meta ON', color='#1e88e5', alpha=0.85)
    ax2.bar(x + 0.15, base_ratios, 0.3, label='Baseline', color='#90caf9', alpha=0.6)
    ax2.set_title('Environment Allocation Ratio (%)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels([s.capitalize() for s in svo_names], rotation=15)
    ax2.set_ylabel('Env %'); ax2.legend(); ax2.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Fig 36: Multi-Resource ATE & Allocation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig36_multi_resource_ate.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[N3] Fig 36 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [N3] Multi-Resource 환경 실험")
    print("=" * 60)

    results = run_experiment()

    print("\n--- Multi-Resource Summary ---")
    for svo_name in SVO_CONDITIONS:
        d = results[svo_name]
        print(f"  {svo_name:>12s} | Meta W={d['meta_welfare']:.1f} | Base W={d['base_welfare']:.1f} | ATE={d['ate_welfare']:+.1f} | Env%={d['meta_env_ratio']*100:.1f}")

    plot_fig35(results)
    plot_fig36(results)

    json_data = {sn: {k: v for k, v in d.items() if k not in ("meta_runs", "base_runs")} for sn, d in results.items()}
    json_path = os.path.join(OUTPUT_DIR, "multi_resource_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\n[N3] 결과 JSON: {json_path}")
