"""
O1: 기후 협상 게임 환경 (Climate Negotiation Game)
EthicaAI Phase O — 연구축 II: 현실 사회 딜레마

N개 국가가 탄소 감축 비용을 배분하는 다자간 협상 게임.
각 국가는 GDP, 인구, 탄소 배출량이 다르며,
메타랭킹(λ)이 공정한 비용 배분에 미치는 효과를 분석합니다.

출력: Fig 39 (국가별 감축 기여 + λ 효과), Fig 40 (기후 협상 ATE)
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

# === 국가 설정 (하드코딩 금지: 설정 딕셔너리로 분리) ===
COUNTRY_CONFIGS = {
    "Country_A": {"gdp": 21.0, "population": 331, "emission": 5.0, "name": "High-GDP High-Emit"},
    "Country_B": {"gdp": 14.7, "population": 1412, "emission": 10.0, "name": "Mid-GDP Highest-Emit"},
    "Country_C": {"gdp": 5.1, "population": 126, "emission": 1.1, "name": "High-GDP Low-Emit"},
    "Country_D": {"gdp": 3.9, "population": 83, "emission": 0.7, "name": "High-GDP Low-Emit EU"},
    "Country_E": {"gdp": 2.7, "population": 1380, "emission": 2.6, "name": "Low-GDP High-Pop"},
    "Country_F": {"gdp": 0.5, "population": 220, "emission": 0.2, "name": "Low-GDP Low-Emit"},
}

# 기후 파라미터 (설정 분리)
CLIMATE_PARAMS = {
    "global_carbon_budget": 20.0,       # Gt CO2 (남은 탄소 예산)
    "damage_per_degree": 2.5,            # GDP % 손실 per °C
    "abatement_cost_factor": 0.02,       # 감축 비용 계수
    "cooperation_multiplier": 1.8,       # 집단 감축의 시너지 효과
    "free_rider_penalty": 0.3,           # 무임승차 시 명성 손실
}

SVO_CONDITIONS = {"selfish": 0.0, "individualist": 15.0, "prosocial": 45.0, "altruistic": 90.0}
N_ROUNDS = 100
N_SEEDS = 10


def simulate_climate_negotiation(svo_theta_deg, seed, use_meta=True):
    """기후 협상 시뮬레이션"""
    rng = np.random.RandomState(seed)
    svo_theta = np.radians(svo_theta_deg)
    lambda_base = np.sin(svo_theta)
    n_countries = len(COUNTRY_CONFIGS)
    countries = list(COUNTRY_CONFIGS.keys())

    # 상태 초기화
    emissions = np.array([COUNTRY_CONFIGS[c]["emission"] for c in countries])
    gdps = np.array([COUNTRY_CONFIGS[c]["gdp"] for c in countries])
    populations = np.array([COUNTRY_CONFIGS[c]["population"] for c in countries])

    # 기록
    reduction_history = np.zeros((N_ROUNDS, n_countries))
    welfare_history = np.zeros((N_ROUNDS, n_countries))
    global_temp = np.zeros(N_ROUNDS)
    lambda_history = np.zeros((N_ROUNDS, n_countries))

    remaining_budget = CLIMATE_PARAMS["global_carbon_budget"]

    for t in range(N_ROUNDS):
        # 국가별 λ 결정
        lambdas = np.zeros(n_countries)
        for i in range(n_countries):
            if use_meta:
                # 동적 λ: GDP 대비 기후 피해 비율에 따라 조절
                climate_risk = emissions[i] / (gdps[i] + 1e-8) * 10
                budget_urgency = max(0, 1 - remaining_budget / CLIMATE_PARAMS["global_carbon_budget"])

                if climate_risk > 0.5 or budget_urgency > 0.7:
                    lambdas[i] = min(1.0, lambda_base * 1.5)  # 위기감 → 협력 증가
                elif gdps[i] < 1.0:
                    lambdas[i] = max(0, lambda_base * 0.5)  # 저소득국 → 여유 없음
                else:
                    lambdas[i] = lambda_base
            else:
                lambdas[i] = lambda_base

        lambda_history[t] = lambdas

        # 감축 결정 (GDP의 일정 비율을 감축에 투자)
        reductions = np.zeros(n_countries)
        for i in range(n_countries):
            base_effort = 0.01 + lambdas[i] * 0.04  # GDP의 1~5%
            noise = rng.normal(0, 0.005)
            reductions[i] = np.clip(base_effort, 0, 0.1) * gdps[i]
            reductions[i] += noise

        reduction_history[t] = reductions

        # 글로벌 효과
        total_reduction = np.sum(reductions)
        synergy = total_reduction * CLIMATE_PARAMS["cooperation_multiplier"]

        # 탄소 예산 소진
        net_emission = np.sum(emissions) - synergy * 0.1
        remaining_budget -= net_emission * 0.01
        remaining_budget = max(0, remaining_budget)

        # 온도 상승
        temp_rise = (1 - remaining_budget / CLIMATE_PARAMS["global_carbon_budget"]) * 3.0
        global_temp[t] = temp_rise

        # 복지 계산 (GDP - 감축비용 - 기후피해)
        for i in range(n_countries):
            abatement_cost = reductions[i] * CLIMATE_PARAMS["abatement_cost_factor"] * 10
            climate_damage = gdps[i] * CLIMATE_PARAMS["damage_per_degree"] / 100 * temp_rise

            # 무임승차 페널티 (감축 적은 국가)
            avg_effort = np.mean(reductions / (gdps + 1e-8))
            own_effort = reductions[i] / (gdps[i] + 1e-8)
            reputation = max(0, 1 - CLIMATE_PARAMS["free_rider_penalty"] * max(0, avg_effort - own_effort) * 10)

            welfare_history[t, i] = gdps[i] - abatement_cost - climate_damage * reputation
            gdps[i] = max(0.1, welfare_history[t, i])  # GDP 업데이트

    return {
        "reductions": reduction_history,
        "welfare": welfare_history,
        "global_temp": global_temp,
        "lambdas": lambda_history,
        "final_reduction": float(np.mean(np.sum(reduction_history[-20:], axis=1))),
        "final_temp": float(np.mean(global_temp[-20:])),
        "final_welfare": float(np.mean(np.sum(welfare_history[-20:], axis=1))),
        "gini_welfare": float(_gini(np.mean(welfare_history[-20:], axis=0))),
    }


def _gini(x):
    x = np.abs(x)
    if np.sum(x) == 0:
        return 0
    sorted_x = np.sort(x)
    n = len(sorted_x)
    cum = np.cumsum(sorted_x)
    return float((2 * np.sum(np.arange(1, n+1) * sorted_x)) / (n * np.sum(sorted_x) + 1e-8) - (n+1)/n)


def run_experiment():
    results = {}
    for svo_name, svo_theta in SVO_CONDITIONS.items():
        meta_runs = [simulate_climate_negotiation(svo_theta, s, True) for s in range(N_SEEDS)]
        base_runs = [simulate_climate_negotiation(svo_theta, s, False) for s in range(N_SEEDS)]
        results[svo_name] = {
            "meta_reduction": float(np.mean([r["final_reduction"] for r in meta_runs])),
            "base_reduction": float(np.mean([r["final_reduction"] for r in base_runs])),
            "meta_temp": float(np.mean([r["final_temp"] for r in meta_runs])),
            "base_temp": float(np.mean([r["final_temp"] for r in base_runs])),
            "meta_welfare": float(np.mean([r["final_welfare"] for r in meta_runs])),
            "base_welfare": float(np.mean([r["final_welfare"] for r in base_runs])),
            "ate_reduction": float(np.mean([r["final_reduction"] for r in meta_runs]) -
                                   np.mean([r["final_reduction"] for r in base_runs])),
            "ate_temp": float(np.mean([r["final_temp"] for r in meta_runs]) -
                              np.mean([r["final_temp"] for r in base_runs])),
            "meta_gini": float(np.mean([r["gini_welfare"] for r in meta_runs])),
            "base_gini": float(np.mean([r["gini_welfare"] for r in base_runs])),
            "meta_runs": meta_runs,
            "base_runs": base_runs,
        }
    return results


def plot_fig39(results):
    """Fig 39: 기후 협상 — 국가별 감축 + 온도 경로"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fig 39: Climate Negotiation — Meta-Ranking Effect on Global Cooperation",
                 fontsize=14, fontweight='bold', y=0.98)
    colors = {'selfish': '#e53935', 'individualist': '#ff9800', 'prosocial': '#1e88e5', 'altruistic': '#43a047'}

    # 상단 좌: 온도 경로
    ax = axes[0, 0]
    for svo_name in SVO_CONDITIONS:
        meta_run = results[svo_name]["meta_runs"][0]
        base_run = results[svo_name]["base_runs"][0]
        ax.plot(meta_run["global_temp"], color=colors[svo_name], linewidth=2, label=f'{svo_name} (Meta)')
        ax.plot(base_run["global_temp"], color=colors[svo_name], linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(1.5, color='red', linestyle=':', alpha=0.7, label='Paris 1.5°C Target')
    ax.set_title('Global Temperature Rise (°C)', fontweight='bold')
    ax.set_xlabel('Round'); ax.set_ylabel('°C'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 상단 우: 총 감축량
    ax = axes[0, 1]
    for svo_name in SVO_CONDITIONS:
        meta_run = results[svo_name]["meta_runs"][0]
        base_run = results[svo_name]["base_runs"][0]
        ax.plot(np.sum(meta_run["reductions"], axis=1), color=colors[svo_name], linewidth=2, label=f'{svo_name} (Meta)')
        ax.plot(np.sum(base_run["reductions"], axis=1), color=colors[svo_name], linewidth=1, linestyle='--', alpha=0.5)
    ax.set_title('Total Global Reduction Effort', fontweight='bold')
    ax.set_xlabel('Round'); ax.set_ylabel('Effort'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 하단 좌: 국가별 감축 (prosocial Meta)
    ax = axes[1, 0]
    meta_run = results["prosocial"]["meta_runs"][0]
    country_names = [COUNTRY_CONFIGS[c]["name"][:10] for c in COUNTRY_CONFIGS]
    for i, name in enumerate(country_names):
        ax.plot(meta_run["reductions"][:, i], linewidth=1.5, label=name)
    ax.set_title('Country-level Reduction (Prosocial Meta)', fontweight='bold')
    ax.set_xlabel('Round'); ax.set_ylabel('Effort'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 하단 우: λ 분화(prosocial Meta)
    ax = axes[1, 1]
    for i, name in enumerate(country_names):
        ax.plot(meta_run["lambdas"][:, i], linewidth=1.5, label=name)
    ax.set_title('Dynamic λ by Country (Prosocial)', fontweight='bold')
    ax.set_xlabel('Round'); ax.set_ylabel('λ'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, "fig39_climate_negotiation.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[O1] Fig 39 저장: {path}")


def plot_fig40(results):
    """Fig 40: 기후 협상 ATE + 공정성"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Fig 40: Climate Negotiation ATE — Reduction, Temperature, Equity",
                 fontsize=14, fontweight='bold', y=1.02)

    svo_names = list(SVO_CONDITIONS.keys())
    x = np.arange(len(svo_names))

    # ATE Reduction
    ax = axes[0]
    ates = [results[s]["ate_reduction"] for s in svo_names]
    colors = ['#43a047' if v > 0 else '#e53935' for v in ates]
    ax.bar(x, ates, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_title('ATE: Reduction Effort', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([s.cap() if hasattr(s, 'cap') else s.capitalize() for s in svo_names], rotation=15, fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5); ax.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(ates):
        ax.text(i, v + 0.002 * np.sign(v), f'{v:+.3f}', ha='center', fontweight='bold', fontsize=9)

    # ATE Temperature
    ax = axes[1]
    ates_t = [results[s]["ate_temp"] for s in svo_names]
    colors_t = ['#43a047' if v < 0 else '#e53935' for v in ates_t]  # 온도 감소가 좋음
    ax.bar(x, ates_t, color=colors_t, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_title('ATE: Temperature (°C, lower=better)', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in svo_names], rotation=15, fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5); ax.grid(True, axis='y', alpha=0.3)

    # Gini 비교
    ax = axes[2]
    meta_g = [results[s]["meta_gini"] for s in svo_names]
    base_g = [results[s]["base_gini"] for s in svo_names]
    ax.bar(x - 0.15, meta_g, 0.3, label='Meta ON', color='#1e88e5', alpha=0.85)
    ax.bar(x + 0.15, base_g, 0.3, label='Baseline', color='#90caf9', alpha=0.6)
    ax.set_title('Welfare Gini (lower=fairer)', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in svo_names], rotation=15, fontsize=9)
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig40_climate_ate.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[O1] Fig 40 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [O1] 기후 협상 게임 환경")
    print("=" * 60)

    results = run_experiment()

    print("\n--- Climate Negotiation Summary ---")
    for svo_name in SVO_CONDITIONS:
        d = results[svo_name]
        print(f"  {svo_name:>12s} | Meta Red={d['meta_reduction']:.3f} | Base Red={d['base_reduction']:.3f} | "
              f"ATE={d['ate_reduction']:+.3f} | Temp Meta={d['meta_temp']:.2f}°C")

    plot_fig39(results)
    plot_fig40(results)

    json_data = {sn: {k: v for k, v in d.items() if k not in ("meta_runs", "base_runs")} for sn, d in results.items()}
    json_path = os.path.join(OUTPUT_DIR, "climate_negotiation_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\n[O1] 결과 JSON: {json_path}")
