"""
Q6: 정책 시사점 리포트
EthicaAI Phase Q — 실용적 가치

연구 결과의 정책적 시사점을 정량화:
1. AI 규제 정책: 메타랭킹 기반 AI 규제 프레임워크 시뮬레이션
2. 공공재 제공: 세금/복지 최적 배분 모델
3. 기후 정책: 탄소 배출권 거래 설계

출력: Fig 67 (정책 시뮬레이션), Fig 68 (정책 권고 대시보드)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json, os, sys

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "ETHICAAI_OUTPUT_DIR", "simulation/outputs/reproduce")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_STEPS = 200
N_SEEDS = 10


def ai_regulation_sim(regulation_level, meta_fraction, seed):
    """AI 규제 정책 시뮬레이션"""
    rng = np.random.RandomState(seed)
    n_companies = 50
    
    innovation = np.full(n_companies, 50.0)
    safety = np.full(n_companies, 50.0)
    public_trust = 50.0
    
    innovate_hist, safety_hist, trust_hist = [], [], []
    
    for t in range(N_STEPS):
        n_meta = int(n_companies * meta_fraction)
        
        for i in range(n_companies):
            if i < n_meta:
                # 메타랭킹: 공공 신뢰에 반응하여 안전/혁신 균형
                if public_trust < 30:
                    safety_weight = 0.8
                elif public_trust > 70:
                    safety_weight = 0.3
                else:
                    safety_weight = 0.5
            else:
                safety_weight = regulation_level
            
            innovation[i] += rng.normal(1.0 * (1 - safety_weight), 2)
            safety[i] += rng.normal(1.0 * safety_weight, 1)
            
            innovation[i] = np.clip(innovation[i], 0, 100)
            safety[i] = np.clip(safety[i], 0, 100)
        
        avg_safety = np.mean(safety)
        avg_innovation = np.mean(innovation)
        incident_prob = max(0, 0.02 * (100 - avg_safety) / 100)
        incident = rng.random() < incident_prob
        
        if incident:
            public_trust = max(0, public_trust - rng.uniform(5, 15))
        else:
            public_trust = min(100, public_trust + 0.5)
        
        innovate_hist.append(float(avg_innovation))
        safety_hist.append(float(avg_safety))
        trust_hist.append(float(public_trust))
    
    return {
        "final_innovation": float(np.mean(innovate_hist[-20:])),
        "final_safety": float(np.mean(safety_hist[-20:])),
        "final_trust": float(np.mean(trust_hist[-20:])),
        "composite": float(np.mean(innovate_hist[-20:]) * 0.4 + np.mean(safety_hist[-20:]) * 0.3 + np.mean(trust_hist[-20:]) * 0.3),
        "trust_history": trust_hist,
    }


def carbon_tax_sim(tax_rate, meta_fraction, seed):
    """탄소세 정책 시뮬레이션"""
    rng = np.random.RandomState(seed)
    n_firms = 30
    
    emissions = np.full(n_firms, 100.0)
    profit = np.full(n_firms, 50.0)
    total_emissions_hist = []
    total_profit_hist = []
    
    for t in range(N_STEPS):
        n_meta = int(n_firms * meta_fraction)
        
        for i in range(n_firms):
            if i < n_meta:
                global_emissions = np.mean(emissions)
                if global_emissions > 70:
                    reduction_effort = 0.8
                elif global_emissions < 30:
                    reduction_effort = 0.2
                else:
                    reduction_effort = 0.5
            else:
                reduction_effort = tax_rate
            
            emissions[i] *= (1 - 0.01 * reduction_effort + rng.normal(0, 0.005))
            cost = reduction_effort * 2
            revenue_loss = max(0, (100 - emissions[i]) * 0.01)
            tax_paid = emissions[i] * tax_rate * 0.01
            profit[i] = max(0, profit[i] - cost - tax_paid + rng.normal(1, 0.5))
            
            emissions[i] = np.clip(emissions[i], 0, 200)
        
        total_emissions_hist.append(float(np.sum(emissions)))
        total_profit_hist.append(float(np.mean(profit)))
    
    return {
        "final_emissions": float(np.mean(total_emissions_hist[-20:])),
        "final_profit": float(np.mean(total_profit_hist[-20:])),
        "emission_reduction": float(1 - np.mean(total_emissions_hist[-20:]) / total_emissions_hist[0]),
    }


def run_policy_analysis():
    reg_levels = np.linspace(0, 1, 11)
    meta_fractions = [0.0, 0.1, 0.3, 0.5]
    
    ai_results = {}
    for mf in meta_fractions:
        ai_results[mf] = {}
        for rl in reg_levels:
            runs = [ai_regulation_sim(rl, mf, s) for s in range(N_SEEDS)]
            ai_results[mf][float(rl)] = {
                "innovation": float(np.mean([r["final_innovation"] for r in runs])),
                "safety": float(np.mean([r["final_safety"] for r in runs])),
                "trust": float(np.mean([r["final_trust"] for r in runs])),
                "composite": float(np.mean([r["composite"] for r in runs])),
            }
    
    carbon_results = {}
    tax_rates = np.linspace(0, 0.5, 11)
    for mf in meta_fractions:
        carbon_results[mf] = {}
        for tr in tax_rates:
            runs = [carbon_tax_sim(tr, mf, s) for s in range(N_SEEDS)]
            carbon_results[mf][float(tr)] = {
                "emissions": float(np.mean([r["final_emissions"] for r in runs])),
                "profit": float(np.mean([r["final_profit"] for r in runs])),
                "reduction": float(np.mean([r["emission_reduction"] for r in runs])),
            }
    
    return ai_results, carbon_results


def plot_fig67(ai_results, carbon_results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Fig 67: Policy Simulation — Meta-Ranking in AI Regulation & Carbon Tax",
                 fontsize=14, fontweight='bold', y=0.98)
    
    mf_colors = {0.0: '#e53935', 0.1: '#ff9800', 0.3: '#1e88e5', 0.5: '#43a047'}
    reg_levels = sorted(next(iter(ai_results.values())).keys())
    
    # 상단: AI 규제
    for idx, metric in enumerate(["innovation", "safety", "composite"]):
        ax = axes[0, idx]
        for mf, color in mf_colors.items():
            vals = [ai_results[mf][rl][metric] for rl in reg_levels]
            ax.plot(reg_levels, vals, 'o-', color=color, linewidth=2, label=f'Meta={mf:.0%}')
        ax.set_title(f'AI {metric.capitalize()} vs Regulation', fontweight='bold')
        ax.set_xlabel('Regulation Level'); ax.set_ylabel(metric.capitalize())
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # 하단: 탄소세
    tax_rates = sorted(next(iter(carbon_results.values())).keys())
    for idx, metric in enumerate(["emissions", "profit", "reduction"]):
        ax = axes[1, idx]
        for mf, color in mf_colors.items():
            vals = [carbon_results[mf][tr][metric] for tr in tax_rates]
            ax.plot(tax_rates, vals, 'o-', color=color, linewidth=2, label=f'Meta={mf:.0%}')
        label = metric.capitalize()
        if metric == "reduction":
            label = "Emission Reduction (%)"
        ax.set_title(f'Carbon {label} vs Tax Rate', fontweight='bold')
        ax.set_xlabel('Tax Rate'); ax.set_ylabel(label)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, "fig67_policy_simulation.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[Q6] Fig 67 저장: {path}")


def plot_fig68(ai_results, carbon_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Fig 68: Policy Recommendation Dashboard — Optimal Meta-Ranking Fraction",
                 fontsize=14, fontweight='bold', y=1.02)
    
    meta_fractions = sorted(ai_results.keys())
    
    # 좌: AI 최적 규제 수준
    ax = axes[0]
    for mf in meta_fractions:
        composites = ai_results[mf]
        best_reg = max(composites.items(), key=lambda x: x[1]["composite"])
        ax.bar(f'Meta={mf:.0%}', best_reg[1]["composite"], 
              color='#1e88e5' if mf > 0 else '#e53935', alpha=0.8, edgecolor='black')
        ax.text(f'Meta={mf:.0%}', best_reg[1]["composite"] + 0.5,
               f'Reg={best_reg[0]:.1f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_title('Best AI Regulation Composite Score', fontweight='bold')
    ax.set_ylabel('Composite (40% Innovation + 30% Safety + 30% Trust)')
    ax.grid(True, axis='y', alpha=0.3)
    
    # 우: 탄소세 최적점
    ax = axes[1]
    for mf in meta_fractions:
        reductions = carbon_results[mf]
        best_tax = max(reductions.items(), key=lambda x: x[1]["reduction"] - x[1]["emissions"] * 0.001)
        ax.bar(f'Meta={mf:.0%}', best_tax[1]["reduction"] * 100,
              color='#43a047' if mf > 0 else '#e53935', alpha=0.8, edgecolor='black')
        ax.text(f'Meta={mf:.0%}', best_tax[1]["reduction"] * 100 + 0.5,
               f'Tax={best_tax[0]:.2f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_title('Best Carbon Tax Emission Reduction', fontweight='bold')
    ax.set_ylabel('Emission Reduction (%)'); ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig68_policy_dashboard.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[Q6] Fig 68 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [Q6] 정책 시사점 리포트")
    print("=" * 60)
    ai_res, carbon_res = run_policy_analysis()
    
    print("\n--- AI 규제 최적점 ---")
    for mf in sorted(ai_res.keys()):
        best = max(ai_res[mf].items(), key=lambda x: x[1]["composite"])
        print(f"  Meta={mf:.0%}: 최적 규제={best[0]:.1f}, Composite={best[1]['composite']:.1f}")
    
    print("\n--- 탄소세 최적점 ---")
    for mf in sorted(carbon_res.keys()):
        best = max(carbon_res[mf].items(), key=lambda x: x[1]["reduction"])
        print(f"  Meta={mf:.0%}: 최적 세율={best[0]:.2f}, 감축={best[1]['reduction']*100:.1f}%")
    
    plot_fig67(ai_res, carbon_res)
    plot_fig68(ai_res, carbon_res)
    
    json_path = os.path.join(OUTPUT_DIR, "policy_results.json")
    with open(json_path, 'w') as f:
        json.dump({"ai_regulation": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in ai_res.items()},
                   "carbon_tax": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in carbon_res.items()}}, f, indent=2)
    print(f"\n[Q6] JSON: {json_path}")
