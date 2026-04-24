"""
O2: 백신 배분 딜레마 (Vaccine Allocation Dilemma)
EthicaAI Phase O — 연구축 II: 현실 사회 딜레마

N개 지역이 한정된 백신을 배분하는 공공재 딜레마.
메타랭킹(λ)이 공정한 배분에 미치는 효과를 분석합니다.

출력: Fig 41 (지역별 배분 + 사망률), Fig 42 (ATE)
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

# === 지역 설정 (하드코딩 금지) ===
REGION_CONFIGS = {
    "region_A": {"population": 10_000_000, "infection_rate": 0.05, "hospital_cap": 0.08, "name": "Low-Risk Rich"},
    "region_B": {"population": 50_000_000, "infection_rate": 0.15, "hospital_cap": 0.03, "name": "High-Risk Dense"},
    "region_C": {"population": 5_000_000,  "infection_rate": 0.02, "hospital_cap": 0.12, "name": "Low-Risk Small"},
    "region_D": {"population": 30_000_000, "infection_rate": 0.10, "hospital_cap": 0.05, "name": "Mid-Risk Large"},
    "region_E": {"population": 20_000_000, "infection_rate": 0.08, "hospital_cap": 0.06, "name": "Mid-Risk Mid"},
}

VACCINE_PARAMS = {
    "total_doses_ratio": 0.3,        # 전체 인구 대비 30%만 백신 가용
    "efficacy": 0.85,
    "mortality_untreated": 0.02,
    "mortality_vaccinated": 0.002,
    "herd_immunity_threshold": 0.7,
}

SVO_CONDITIONS = {"selfish": 0.0, "individualist": 15.0, "prosocial": 45.0, "altruistic": 90.0}
N_ROUNDS = 80
N_SEEDS = 10


def jain_fairness(x):
    """Jain의 공정성 지수: 1=완전공정"""
    x = np.abs(x) + 1e-8
    n = len(x)
    return float((np.sum(x)) ** 2 / (n * np.sum(x ** 2)))


def simulate_vaccine(svo_theta_deg, seed, use_meta=True):
    """백신 배분 시뮬레이션"""
    rng = np.random.RandomState(seed)
    svo_theta = np.radians(svo_theta_deg)
    lambda_base = np.sin(svo_theta)
    
    regions = list(REGION_CONFIGS.keys())
    n_regions = len(regions)
    populations = np.array([REGION_CONFIGS[r]["population"] for r in regions], dtype=float)
    infection_rates = np.array([REGION_CONFIGS[r]["infection_rate"] for r in regions])
    hospital_caps = np.array([REGION_CONFIGS[r]["hospital_cap"] for r in regions])
    
    total_pop = np.sum(populations)
    total_doses = int(total_pop * VACCINE_PARAMS["total_doses_ratio"])
    
    # 기록
    allocation_history = np.zeros((N_ROUNDS, n_regions))
    deaths_prevented = np.zeros(N_ROUNDS)
    fairness_history = np.zeros(N_ROUNDS)
    lambda_history = np.zeros((N_ROUNDS, n_regions))
    vaccinated = np.zeros(n_regions)
    
    for t in range(N_ROUNDS):
        # 잔여 백신
        remaining_doses = total_doses - np.sum(vaccinated)
        if remaining_doses <= 0:
            allocation_history[t:] = 0
            break
        
        doses_this_round = min(remaining_doses, total_doses / N_ROUNDS * 1.5)
        
        # 각 지역의 λ 결정
        lambdas = np.zeros(n_regions)
        for i in range(n_regions):
            if use_meta:
                # 위기 기반 λ 조절
                vax_coverage = vaccinated[i] / (populations[i] + 1e-8)
                crisis = infection_rates[i] * (1 - vax_coverage) / (hospital_caps[i] + 1e-8)
                
                if crisis > 2.0:
                    lambdas[i] = min(1.0, lambda_base * 0.3)  # 자기 지역 위기 → 이기적
                elif vax_coverage > VACCINE_PARAMS["herd_immunity_threshold"]:
                    lambdas[i] = min(1.0, lambda_base * 1.8)  # 여유 → 관대
                else:
                    lambdas[i] = lambda_base
            else:
                lambdas[i] = lambda_base
        
        lambda_history[t] = lambdas
        
        # 배분 결정 (λ가 높을수록 감염률 비례 배분, 낮을수록 자기 인구 비례)
        claims = np.zeros(n_regions)
        for i in range(n_regions):
            selfish_claim = populations[i] / total_pop  # 인구 비례
            fair_claim = infection_rates[i] / np.sum(infection_rates)  # 감염률 비례 (공정)
            claims[i] = (1 - lambdas[i]) * selfish_claim + lambdas[i] * fair_claim
        
        claims = claims / (np.sum(claims) + 1e-8)
        allocations = claims * doses_this_round + rng.normal(0, doses_this_round * 0.01, n_regions)
        allocations = np.clip(allocations, 0, doses_this_round)
        allocations = allocations / (np.sum(allocations) + 1e-8) * doses_this_round
        
        allocation_history[t] = allocations
        vaccinated += allocations
        
        # 사망 방지 계산
        for i in range(n_regions):
            vax_cov = vaccinated[i] / (populations[i] + 1e-8)
            deaths_base = populations[i] * infection_rates[i] * VACCINE_PARAMS["mortality_untreated"]
            deaths_vax = populations[i] * infection_rates[i] * (
                (1 - vax_cov) * VACCINE_PARAMS["mortality_untreated"] +
                vax_cov * VACCINE_PARAMS["mortality_vaccinated"]
            )
            deaths_prevented[t] += (deaths_base - deaths_vax)
        
        # 공정성 (접종률 기준)
        vax_coverages = vaccinated / (populations + 1e-8)
        fairness_history[t] = jain_fairness(vax_coverages)
    
    return {
        "allocations": allocation_history,
        "deaths_prevented": deaths_prevented,
        "fairness": fairness_history,
        "lambdas": lambda_history,
        "total_deaths_prevented": float(np.sum(deaths_prevented)),
        "final_fairness": float(np.mean(fairness_history[-20:])),
        "final_coverage": float(np.mean(vaccinated / (populations + 1e-8))),
    }


def run_experiment():
    results = {}
    for svo_name, svo_theta in SVO_CONDITIONS.items():
        meta_runs = [simulate_vaccine(svo_theta, s, True) for s in range(N_SEEDS)]
        base_runs = [simulate_vaccine(svo_theta, s, False) for s in range(N_SEEDS)]
        results[svo_name] = {
            "meta_deaths_prevented": float(np.mean([r["total_deaths_prevented"] for r in meta_runs])),
            "base_deaths_prevented": float(np.mean([r["total_deaths_prevented"] for r in base_runs])),
            "meta_fairness": float(np.mean([r["final_fairness"] for r in meta_runs])),
            "base_fairness": float(np.mean([r["final_fairness"] for r in base_runs])),
            "ate_deaths": float(np.mean([r["total_deaths_prevented"] for r in meta_runs]) -
                                np.mean([r["total_deaths_prevented"] for r in base_runs])),
            "ate_fairness": float(np.mean([r["final_fairness"] for r in meta_runs]) -
                                  np.mean([r["final_fairness"] for r in base_runs])),
            "meta_runs": meta_runs,
            "base_runs": base_runs,
        }
    return results


def plot_fig41(results):
    """Fig 41: 백신 배분 — 지역별 배분 + 공정성"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fig 41: Vaccine Allocation — Meta-Ranking Effect on Fair Distribution",
                 fontsize=14, fontweight='bold', y=0.98)
    colors = {'selfish': '#e53935', 'individualist': '#ff9800', 'prosocial': '#1e88e5', 'altruistic': '#43a047'}
    
    # 상좌: 누적 사망 방지
    ax = axes[0, 0]
    for svo in SVO_CONDITIONS:
        meta_run = results[svo]["meta_runs"][0]
        ax.plot(np.cumsum(meta_run["deaths_prevented"]), color=colors[svo], linewidth=2, label=f'{svo} (Meta)')
        base_run = results[svo]["base_runs"][0]
        ax.plot(np.cumsum(base_run["deaths_prevented"]), color=colors[svo], linewidth=1, linestyle='--', alpha=0.5)
    ax.set_title('Cumulative Deaths Prevented', fontweight='bold')
    ax.set_xlabel('Round'); ax.set_ylabel('Lives Saved'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # 상우: Jain 공정성 지수
    ax = axes[0, 1]
    for svo in SVO_CONDITIONS:
        meta_run = results[svo]["meta_runs"][0]
        ax.plot(meta_run["fairness"], color=colors[svo], linewidth=2, label=f'{svo} (Meta)')
        base_run = results[svo]["base_runs"][0]
        ax.plot(base_run["fairness"], color=colors[svo], linewidth=1, linestyle='--', alpha=0.5)
    ax.set_title("Jain's Fairness Index (1=Perfect)", fontweight='bold')
    ax.set_xlabel('Round'); ax.set_ylabel('Fairness'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # 하좌: 지역별 최종 배분 (prosocial Meta)
    ax = axes[1, 0]
    meta_run = results["prosocial"]["meta_runs"][0]
    base_run = results["prosocial"]["base_runs"][0]
    region_names = [REGION_CONFIGS[r]["name"][:12] for r in REGION_CONFIGS]
    x = np.arange(len(region_names))
    meta_alloc = np.sum(meta_run["allocations"], axis=0)
    base_alloc = np.sum(base_run["allocations"], axis=0)
    ax.bar(x - 0.15, meta_alloc / 1e6, 0.3, label='Meta ON', color='#1e88e5')
    ax.bar(x + 0.15, base_alloc / 1e6, 0.3, label='Baseline', color='#90caf9')
    ax.set_title('Total Allocation by Region (Prosocial, M doses)', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(region_names, rotation=20, fontsize=8)
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    
    # 하우: λ 분화 (prosocial Meta)
    ax = axes[1, 1]
    for i, name in enumerate(region_names):
        ax.plot(meta_run["lambdas"][:, i], linewidth=1.5, label=name)
    ax.set_title('Dynamic λ by Region (Prosocial)', fontweight='bold')
    ax.set_xlabel('Round'); ax.set_ylabel('λ'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, "fig41_vaccine_allocation.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[O2] Fig 41 저장: {path}")


def plot_fig42(results):
    """Fig 42: 백신 배분 ATE"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Fig 42: Vaccine Allocation ATE — Deaths Prevented, Fairness, Coverage",
                 fontsize=14, fontweight='bold', y=1.02)
    
    svo_names = list(SVO_CONDITIONS.keys())
    x = np.arange(len(svo_names))
    labels = [s.capitalize() for s in svo_names]
    
    # ATE Deaths Prevented
    ax = axes[0]
    ates = [results[s]["ate_deaths"] for s in svo_names]
    colors_bar = ['#43a047' if v > 0 else '#e53935' for v in ates]
    ax.bar(x, [a / 1e3 for a in ates], color=colors_bar, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_title('ATE: Deaths Prevented (×1000)', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5); ax.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(ates):
        ax.text(i, v / 1e3 + 0.5 * np.sign(v), f'{v / 1e3:+.1f}k', ha='center', fontweight='bold', fontsize=8)
    
    # ATE Fairness
    ax = axes[1]
    ates_f = [results[s]["ate_fairness"] for s in svo_names]
    colors_f = ['#43a047' if v > 0 else '#e53935' for v in ates_f]
    ax.bar(x, ates_f, color=colors_f, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_title('ATE: Jain Fairness', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5); ax.grid(True, axis='y', alpha=0.3)
    
    # Meta vs Base (복지)
    ax = axes[2]
    meta_v = [results[s]["meta_deaths_prevented"] / 1e3 for s in svo_names]
    base_v = [results[s]["base_deaths_prevented"] / 1e3 for s in svo_names]
    ax.bar(x - 0.15, meta_v, 0.3, label='Meta ON', color='#1e88e5')
    ax.bar(x + 0.15, base_v, 0.3, label='Baseline', color='#90caf9')
    ax.set_title('Total Deaths Prevented (×1000)', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=9)
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig42_vaccine_ate.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[O2] Fig 42 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [O2] 백신 배분 딜레마")
    print("=" * 60)
    
    results = run_experiment()
    
    print("\n--- Vaccine Allocation Summary ---")
    for svo_name in SVO_CONDITIONS:
        d = results[svo_name]
        print(f"  {svo_name:>12s} | Meta Deaths={d['meta_deaths_prevented']:.0f} | "
              f"Base={d['base_deaths_prevented']:.0f} | ATE={d['ate_deaths']:+.0f} | "
              f"Fairness Meta={d['meta_fairness']:.3f}")
    
    plot_fig41(results)
    plot_fig42(results)
    
    json_data = {sn: {k: v for k, v in d.items() if k not in ("meta_runs", "base_runs")} for sn, d in results.items()}
    json_path = os.path.join(OUTPUT_DIR, "vaccine_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\n[O2] 결과 JSON: {json_path}")
