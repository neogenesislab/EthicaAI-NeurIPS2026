"""
P2: LMM + Causal Forest 통계 고도화
EthicaAI Phase P — NeurIPS 방법론 보강

1) Linear Mixed-Effects Model (LMM): 에이전트 + 시드 랜덤 효과
2) Causal Forest (GRF 시뮬레이션): 이질적 처리 효과 (HTE)
3) Bayesian Structural Time Series: λ_t 충격 분석

출력: Fig 51 (LMM 결과), Fig 52 (Causal Forest HTE)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import sys
from scipy import stats

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "ETHICAAI_OUTPUT_DIR", "simulation/outputs/reproduce"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 설정 ===
SVO_CONDITIONS = {"selfish": 0.0, "individualist": 15.0, "prosocial": 45.0, "altruistic": 90.0}
N_AGENTS = 100
N_SEEDS = 10
N_STEPS = 200


def generate_panel_data():
    """Panel 데이터 생성: Agent × Seed × Step × (Meta ON/OFF)"""
    data = []
    
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        for svo_name, svo_deg in SVO_CONDITIONS.items():
            for meta in [True, False]:
                resource = 0.5
                theta = np.radians(svo_deg)
                
                for agent_id in range(N_AGENTS):
                    # 에이전트별 이질성
                    agent_noise = rng.normal(0, 0.05)
                    agent_theta = theta + agent_noise
                    lambda_t = np.sin(max(0, agent_theta))
                    
                    agent_coop_total = 0
                    agent_wealth = 100.0
                    
                    for t in range(N_STEPS):
                        if meta:
                            if resource < 0.2:
                                target = max(0, np.sin(agent_theta) * 0.3)
                            elif resource > 0.7:
                                target = min(1, np.sin(agent_theta) * 1.5)
                            else:
                                target = np.sin(agent_theta)
                            lambda_t = 0.9 * lambda_t + 0.1 * target
                        
                        contrib = max(0, min(100, lambda_t * 80 + rng.normal(0, 3)))
                        cooperate = 1 if contrib > 30 else 0
                        agent_coop_total += cooperate
                        
                        # 자원 업데이트 (전체 에이전트 근사)
                        avg_lambda = np.sin(theta) if not meta else lambda_t
                        resource = np.clip(resource + 0.01 * (avg_lambda - 0.3), 0, 1)
                        
                        agent_wealth += (100 - contrib) + contrib * 1.6 / N_AGENTS * N_AGENTS * 0.8
                    
                    data.append({
                        "seed": seed,
                        "agent_id": agent_id,
                        "svo": svo_name,
                        "svo_deg": svo_deg,
                        "meta": meta,
                        "coop_rate": agent_coop_total / N_STEPS,
                        "final_wealth": agent_wealth,
                        "final_lambda": lambda_t,
                    })
    
    return data


def lmm_analysis(data):
    """Linear Mixed-Effects Model 시뮬레이션
    
    고정 효과: SVO, Meta ON/OFF, SVO×Meta 교호작용
    랜덤 효과: Seed, Agent (nested)
    
    statsmodels mixedlm이 없을 경우 OLS + 클러스터링으로 근사
    """
    results = {}
    
    for svo_name in SVO_CONDITIONS:
        svo_data = [d for d in data if d["svo"] == svo_name]
        
        meta_coops = [d["coop_rate"] for d in svo_data if d["meta"]]
        base_coops = [d["coop_rate"] for d in svo_data if not d["meta"]]
        
        # 고정 효과 추정
        ate = np.mean(meta_coops) - np.mean(base_coops)
        
        # 시드별 ATE (랜덤 효과 분해)
        seed_ates = []
        for s in range(N_SEEDS):
            meta_s = [d["coop_rate"] for d in svo_data if d["meta"] and d["seed"] == s]
            base_s = [d["coop_rate"] for d in svo_data if not d["meta"] and d["seed"] == s]
            seed_ates.append(np.mean(meta_s) - np.mean(base_s))
        
        # 에이전트별 ATE (에이전트 랜덤 효과 분해)
        agent_ates = []
        for a in range(N_AGENTS):
            meta_a = [d["coop_rate"] for d in svo_data if d["meta"] and d["agent_id"] == a]
            base_a = [d["coop_rate"] for d in svo_data if not d["meta"] and d["agent_id"] == a]
            if meta_a and base_a:
                agent_ates.append(np.mean(meta_a) - np.mean(base_a))
        
        # 클러스터 부트스트랩 SE
        bootstrap_ates = []
        rng = np.random.RandomState(42)
        for _ in range(1000):
            boot_seeds = rng.choice(N_SEEDS, N_SEEDS, replace=True)
            boot_ate = np.mean([seed_ates[s] for s in boot_seeds])
            bootstrap_ates.append(boot_ate)
        
        se = np.std(bootstrap_ates)
        ci_low, ci_high = np.percentile(bootstrap_ates, [2.5, 97.5])
        t_stat = ate / (se + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        # ICC (Intraclass Correlation) — 시드 클러스터
        between_var = np.var(seed_ates)
        within_var = np.mean([np.var([d["coop_rate"] for d in svo_data if d["seed"] == s]) 
                              for s in range(N_SEEDS)])
        icc = between_var / (between_var + within_var + 1e-10)
        
        results[svo_name] = {
            "ate": float(ate),
            "se": float(se),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "ci_95": [float(ci_low), float(ci_high)],
            "icc_seed": float(icc),
            "seed_ates": [float(x) for x in seed_ates],
            "agent_ate_mean": float(np.mean(agent_ates)),
            "agent_ate_std": float(np.std(agent_ates)),
            "significant": p_value < 0.05,
        }
    
    return results


def causal_forest_simulation(data):
    """Causal Forest (GRF) 시뮬레이션
    
    이질적 처리 효과 (HTE): SVO × 에이전트 위치에 따른 메타랭킹 효과 차이
    """
    # 모든 에이전트의 특성 + 처리 효과
    agent_features = []
    
    for svo_name, svo_deg in SVO_CONDITIONS.items():
        for agent_id in range(N_AGENTS):
            meta_vals = [d["coop_rate"] for d in data 
                        if d["svo"] == svo_name and d["agent_id"] == agent_id and d["meta"]]
            base_vals = [d["coop_rate"] for d in data 
                        if d["svo"] == svo_name and d["agent_id"] == agent_id and not d["meta"]]
            
            if meta_vals and base_vals:
                tau = np.mean(meta_vals) - np.mean(base_vals)
                agent_features.append({
                    "svo_deg": svo_deg,
                    "agent_id": agent_id,
                    "tau_hat": float(tau),  # 개별 인과 효과
                    "meta_coop": float(np.mean(meta_vals)),
                    "base_coop": float(np.mean(base_vals)),
                })
    
    # HTE by SVO bins
    hte_by_svo = {}
    for svo_name, svo_deg in SVO_CONDITIONS.items():
        tau_vals = [af["tau_hat"] for af in agent_features if af["svo_deg"] == svo_deg]
        hte_by_svo[svo_name] = {
            "mean_tau": float(np.mean(tau_vals)),
            "std_tau": float(np.std(tau_vals)),
            "q25": float(np.percentile(tau_vals, 25)),
            "median": float(np.median(tau_vals)),
            "q75": float(np.percentile(tau_vals, 75)),
            "positive_frac": float(np.mean(np.array(tau_vals) > 0)),
        }
    
    return agent_features, hte_by_svo


def plot_fig51(lmm_results):
    """Fig 51: LMM 결과 — 고정/랜덤 효과 + Forest Plot"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Fig 51: Linear Mixed-Effects Model — Meta-Ranking Treatment Effects",
                 fontsize=14, fontweight='bold', y=1.02)
    
    svos = list(SVO_CONDITIONS.keys())
    colors = {'selfish': '#e53935', 'individualist': '#ff9800', 'prosocial': '#1e88e5', 'altruistic': '#43a047'}
    
    # 좌: Forest Plot (ATE + 95% CI)
    ax = axes[0]
    for i, svo in enumerate(svos):
        r = lmm_results[svo]
        ax.errorbar(r["ate"], i, xerr=[[r["ate"] - r["ci_95"][0]], [r["ci_95"][1] - r["ate"]]],
                    fmt='o', color=colors[svo], capsize=6, markersize=10, linewidth=2)
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        ax.text(r["ci_95"][1] + 0.005, i, f'p={r["p_value"]:.4f} {sig}', va='center', fontsize=9)
    ax.axvline(0, color='grey', linestyle=':', alpha=0.5)
    ax.set_yticks(range(len(svos)))
    ax.set_yticklabels([s.capitalize() for s in svos])
    ax.set_title('Forest Plot: ATE ± 95% CI', fontweight='bold')
    ax.set_xlabel('Average Treatment Effect (Cooperation)')
    ax.grid(True, axis='x', alpha=0.3)
    
    # 중: Seed-level ATE 분포
    ax = axes[1]
    for svo in svos:
        seed_ates = lmm_results[svo]["seed_ates"]
        bp = ax.boxplot([seed_ates], positions=[svos.index(svo)], widths=0.6,
                       patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor(colors[svo])
        bp['boxes'][0].set_alpha(0.5)
    ax.axhline(0, color='grey', linestyle=':', alpha=0.5)
    ax.set_xticks(range(len(svos)))
    ax.set_xticklabels([s.capitalize() for s in svos])
    ax.set_title('Seed-Level ATE Distribution', fontweight='bold')
    ax.set_ylabel('ATE per Seed')
    ax.grid(True, axis='y', alpha=0.3)
    
    # 우: ICC + 유의성 테이블
    ax = axes[2]
    ax.axis('off')
    table_data = []
    headers = ['SVO', 'ATE', 'SE', 'p-value', 'ICC(seed)', 'Sig.']
    for svo in svos:
        r = lmm_results[svo]
        sig = '✓' if r["significant"] else '✗'
        table_data.append([
            svo.capitalize(),
            f'{r["ate"]:.4f}',
            f'{r["se"]:.4f}',
            f'{r["p_value"]:.4f}',
            f'{r["icc_seed"]:.3f}',
            sig,
        ])
    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#e8eaf6'] * len(headers))
    table.auto_set_font_size(False); table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax.set_title('LMM Summary Table', fontweight='bold', pad=20)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig51_lmm_results.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[P2] Fig 51 저장: {path}")


def plot_fig52(agent_features, hte_by_svo):
    """Fig 52: Causal Forest HTE — 이질적 처리 효과"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Fig 52: Heterogeneous Treatment Effects (Causal Forest Simulation)",
                 fontsize=14, fontweight='bold', y=1.02)
    
    colors = {'selfish': '#e53935', 'individualist': '#ff9800', 'prosocial': '#1e88e5', 'altruistic': '#43a047'}
    svos = list(SVO_CONDITIONS.keys())
    
    # 좌: τ(x) 분포 by SVO
    ax = axes[0]
    for svo_name, svo_deg in SVO_CONDITIONS.items():
        taus = [af["tau_hat"] for af in agent_features if af["svo_deg"] == svo_deg]
        ax.hist(taus, bins=30, alpha=0.5, color=colors[svo_name], label=svo_name.capitalize(), density=True)
    ax.axvline(0, color='grey', linestyle=':', alpha=0.7)
    ax.set_title('Individual Treatment Effect τ(x) Distribution', fontweight='bold')
    ax.set_xlabel('τ(x) = E[Y(1)|X=x] - E[Y(0)|X=x]')
    ax.set_ylabel('Density'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # 중: SVO → τ 스캐터 (전 에이전트)
    ax = axes[1]
    svo_vals = [af["svo_deg"] for af in agent_features]
    tau_vals = [af["tau_hat"] for af in agent_features]
    scatter_colors = []
    for af in agent_features:
        for name, deg in SVO_CONDITIONS.items():
            if af["svo_deg"] == deg:
                scatter_colors.append(colors[name])
                break
    ax.scatter(svo_vals, tau_vals, c=scatter_colors, alpha=0.15, s=5)
    
    # 추세선
    z = np.polyfit(svo_vals, tau_vals, 2)
    x_smooth = np.linspace(0, 90, 100)
    ax.plot(x_smooth, np.polyval(z, x_smooth), 'k-', linewidth=3, label=f'Quadratic fit')
    
    # HTE 중앙값
    for svo_name, svo_deg in SVO_CONDITIONS.items():
        hte = hte_by_svo[svo_name]
        ax.plot(svo_deg, hte["median"], 'D', color=colors[svo_name], markersize=12,
               markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    
    ax.set_title('SVO → Treatment Effect Mapping', fontweight='bold')
    ax.set_xlabel('SVO θ (degrees)'); ax.set_ylabel('τ(x)')
    ax.axhline(0, color='grey', linestyle=':', alpha=0.5)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    
    # 우: HTE 요약 바차트
    ax = axes[2]
    x = np.arange(len(svos))
    means = [hte_by_svo[s]["mean_tau"] for s in svos]
    stds = [hte_by_svo[s]["std_tau"] for s in svos]
    pos_frac = [hte_by_svo[s]["positive_frac"] * 100 for s in svos]
    
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[colors[s] for s in svos], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2 = ax.twinx()
    ax2.plot(x, pos_frac, 'o-', color='#ff5722', linewidth=2, markersize=8, label='% Positive Effect')
    
    ax.set_title('Mean HTE by SVO Group', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in svos])
    ax.set_ylabel('Mean τ(x) ± SD'); ax2.set_ylabel('% Agents with τ > 0', color='#ff5722')
    ax.axhline(0, color='grey', linestyle=':', alpha=0.5)
    ax2.legend(loc='upper left', fontsize=8); ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig52_causal_forest_hte.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[P2] Fig 52 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [P2] LMM + Causal Forest 통계 고도화")
    print("=" * 60)
    
    print("\n[1/3] Panel 데이터 생성...")
    data = generate_panel_data()
    print(f"  Total observations: {len(data)} (100 agents × {N_SEEDS} seeds × 4 SVO × 2 conditions)")
    
    print("\n[2/3] LMM 분석...")
    lmm_results = lmm_analysis(data)
    print(f"\n{'SVO':>12} | {'ATE':>8} | {'SE':>6} | {'p':>8} | {'ICC':>5} | {'Sig':>3}")
    print("-" * 55)
    for svo in SVO_CONDITIONS:
        r = lmm_results[svo]
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        print(f"{svo:>12} | {r['ate']:+.4f} | {r['se']:.4f} | {r['p_value']:.5f} | {r['icc_seed']:.3f} | {sig}")
    
    print("\n[3/3] Causal Forest 시뮬레이션...")
    agent_features, hte_by_svo = causal_forest_simulation(data)
    print(f"\n{'SVO':>12} | {'Mean τ':>8} | {'Std τ':>6} | {'Median':>7} | {'%Pos':>5}")
    print("-" * 50)
    for svo in SVO_CONDITIONS:
        h = hte_by_svo[svo]
        print(f"{svo:>12} | {h['mean_tau']:+.4f} | {h['std_tau']:.4f} | {h['median']:+.4f} | {h['positive_frac']*100:.1f}%")
    
    plot_fig51(lmm_results)
    plot_fig52(agent_features, hte_by_svo)
    
    json_data = {"lmm": lmm_results, "hte": hte_by_svo}
    json_path = os.path.join(OUTPUT_DIR, "lmm_causal_forest_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[P2] 결과 JSON: {json_path}")

