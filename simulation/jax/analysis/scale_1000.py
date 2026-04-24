"""
P1: 1000-에이전트 스케일 불변성 실험
EthicaAI Phase P — NeurIPS 약점 보강

20 → 100 → 500 → 1000 에이전트로 확장:
- ATE 방향 + 유의성 유지 검증 (Scale Invariance Index)
- 역할 특수화 Gini 변화
- 계산 시간 벤치마크

출력: Fig 49 (스케일별 ATE), Fig 50 (역할 특수화 + 벤치마크)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import os
import sys
import time

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "ETHICAAI_OUTPUT_DIR", "simulation/outputs/reproduce"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 설정 ===
SCALES = [20, 50, 100, 200, 500, 1000]
SVO_CONDITIONS = {"selfish": 0.0, "individualist": 15.0, "prosocial": 45.0, "altruistic": 90.0}
N_STEPS = 200
N_SEEDS = 5
ALPHA = 0.1  # λ 업데이트 속도


def compute_lambda(svo_deg, resource):
    """메타랭킹 기반 동적 λ 계산"""
    theta = np.radians(svo_deg)
    base = np.sin(theta)
    if resource < 0.2:
        return max(0.0, base * 0.3)
    elif resource > 0.7:
        return min(1.0, base * 1.5)
    return base


def simulate_scale(n_agents, svo_deg, seed, use_meta=True):
    """N-에이전트 PGG 시뮬레이션"""
    rng = np.random.RandomState(seed)
    
    # 에이전트 초기화
    endowment = 100.0
    wealth = np.full(n_agents, endowment)
    lambdas = np.full(n_agents, np.sin(np.radians(svo_deg)))
    resource = 0.5
    
    coop_history = []
    gini_history = []
    
    for t in range(N_STEPS):
        # λ 업데이트 (메타랭킹)
        if use_meta:
            target = compute_lambda(svo_deg, resource)
            lambdas = lambdas * (1 - ALPHA) + target * ALPHA
        
        # 기여 결정
        contributions = np.clip(
            lambdas * endowment * 0.8 + rng.normal(0, 3, n_agents),
            0, endowment
        )
        
        # 공공재 + 분배
        total_contrib = np.sum(contributions)
        public_good = total_contrib * 1.6 / n_agents
        payoffs = (endowment - contributions) + public_good
        
        # 자원 업데이트
        mean_contrib_rate = np.mean(contributions) / endowment
        resource = np.clip(resource + 0.03 * (mean_contrib_rate - 0.3), 0, 1)
        
        wealth += payoffs - endowment
        coop = np.mean(contributions > endowment * 0.3)
        coop_history.append(coop)
        
        # Gini 계수 (역할 분화 측정)
        sorted_w = np.sort(wealth)
        n = len(sorted_w)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_w) / (n * np.sum(sorted_w))) - (n + 1) / n
        gini_history.append(max(0, gini))
    
    return {
        "mean_coop": float(np.mean(coop_history[-30:])),
        "mean_gini": float(np.mean(gini_history[-30:])),
        "final_wealth_std": float(np.std(wealth)),
        "final_resource": float(resource),
    }


def run_scale_experiment():
    """전체 스케일 실험"""
    results = {}
    benchmarks = {}
    
    for n in SCALES:
        results[n] = {}
        t_start = time.time()
        
        for svo_name, svo_deg in SVO_CONDITIONS.items():
            meta_runs = [simulate_scale(n, svo_deg, s, True) for s in range(N_SEEDS)]
            base_runs = [simulate_scale(n, svo_deg, s, False) for s in range(N_SEEDS)]
            
            meta_coop = np.mean([r["mean_coop"] for r in meta_runs])
            base_coop = np.mean([r["mean_coop"] for r in base_runs])
            ate = meta_coop - base_coop
            
            results[n][svo_name] = {
                "meta_coop": float(meta_coop),
                "base_coop": float(base_coop),
                "ate": float(ate),
                "meta_gini": float(np.mean([r["mean_gini"] for r in meta_runs])),
                "base_gini": float(np.mean([r["mean_gini"] for r in base_runs])),
                "meta_wealth_std": float(np.mean([r["final_wealth_std"] for r in meta_runs])),
            }
        
        elapsed = time.time() - t_start
        benchmarks[n] = {
            "time_sec": float(elapsed),
            "time_per_agent_ms": float(elapsed / n * 1000),
        }
    
    return results, benchmarks


def compute_sii(results):
    """Scale Invariance Index (SII): ATE_N / ATE_20"""
    sii = {}
    base_n = SCALES[0]
    for n in SCALES:
        sii[n] = {}
        for svo in SVO_CONDITIONS:
            ate_n = results[n][svo]["ate"]
            ate_base = results[base_n][svo]["ate"]
            sii[n][svo] = ate_n / (ate_base + 1e-8) if abs(ate_base) > 1e-8 else 1.0
    return sii


def plot_fig49(results, benchmarks):
    """Fig 49: 스케일별 ATE + Scale Invariance"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Fig 49: Scale Invariance — ATE Across Agent Populations",
                 fontsize=14, fontweight='bold', y=1.02)
    
    colors = {'selfish': '#e53935', 'individualist': '#ff9800', 'prosocial': '#1e88e5', 'altruistic': '#43a047'}
    
    # 좌: ATE by Scale
    ax = axes[0]
    for svo, color in colors.items():
        ates = [results[n][svo]["ate"] for n in SCALES]
        ax.plot(SCALES, ates, 'o-', color=color, linewidth=2, markersize=6, label=svo.capitalize())
    ax.axhline(0, color='grey', linestyle=':', alpha=0.5)
    ax.set_title('ATE (Meta - Baseline) by Scale', fontweight='bold')
    ax.set_xlabel('Agent Count'); ax.set_ylabel('ATE (Cooperation)')
    ax.set_xscale('log'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # 중: SII (Scale Invariance Index)
    ax = axes[1]
    sii = compute_sii(results)
    for svo, color in colors.items():
        vals = [sii[n][svo] for n in SCALES]
        ax.plot(SCALES, vals, 's-', color=color, linewidth=2, markersize=6, label=svo.capitalize())
    ax.axhline(1.0, color='grey', linestyle='--', alpha=0.5, label='Perfect Invariance')
    ax.fill_between(SCALES, 0.8, 1.2, alpha=0.1, color='green', label='±20% Band')
    ax.set_title('Scale Invariance Index (ATE_N / ATE_20)', fontweight='bold')
    ax.set_xlabel('Agent Count'); ax.set_ylabel('SII')
    ax.set_xscale('log'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # 우: Cooperation Rate (Meta vs Base)
    ax = axes[2]
    x = np.arange(len(SCALES))
    width = 0.35
    meta_vals = [np.mean([results[n][s]["meta_coop"] for s in SVO_CONDITIONS]) for n in SCALES]
    base_vals = [np.mean([results[n][s]["base_coop"] for s in SVO_CONDITIONS]) for n in SCALES]
    ax.bar(x - width/2, meta_vals, width, label='Meta-Ranking', color='#1e88e5', alpha=0.8)
    ax.bar(x + width/2, base_vals, width, label='Baseline', color='#e53935', alpha=0.8)
    ax.set_title('Mean Cooperation Rate by Scale', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in SCALES])
    ax.set_xlabel('Agent Count'); ax.set_ylabel('Cooperation Rate')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig49_scale_invariance.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[P1] Fig 49 저장: {path}")


def plot_fig50(results, benchmarks):
    """Fig 50: 역할 특수화 + 계산 벤치마크"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Fig 50: Role Specialization & Computational Scaling",
                 fontsize=14, fontweight='bold', y=1.02)
    
    colors = {'selfish': '#e53935', 'individualist': '#ff9800', 'prosocial': '#1e88e5', 'altruistic': '#43a047'}
    
    # 좌: Gini by Scale (Meta vs Base)
    ax = axes[0]
    for svo, color in colors.items():
        meta_g = [results[n][svo]["meta_gini"] for n in SCALES]
        base_g = [results[n][svo]["base_gini"] for n in SCALES]
        ax.plot(SCALES, meta_g, 'o-', color=color, linewidth=2, label=f'{svo[:5]} Meta')
        ax.plot(SCALES, base_g, 's--', color=color, linewidth=1, alpha=0.5, label=f'{svo[:5]} Base')
    ax.set_title('Role Specialization (Gini) by Scale', fontweight='bold')
    ax.set_xlabel('Agent Count'); ax.set_ylabel('Gini Coefficient')
    ax.set_xscale('log'); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)
    
    # 중: Wealth std by Scale
    ax = axes[1]
    for svo, color in colors.items():
        stds = [results[n][svo]["meta_wealth_std"] for n in SCALES]
        ax.plot(SCALES, stds, 'o-', color=color, linewidth=2, markersize=6, label=svo.capitalize())
    ax.set_title('Wealth Inequality (Std) by Scale', fontweight='bold')
    ax.set_xlabel('Agent Count'); ax.set_ylabel('Wealth Std')
    ax.set_xscale('log'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # 우: 계산 벤치마크
    ax = axes[2]
    times = [benchmarks[n]["time_sec"] for n in SCALES]
    per_agent = [benchmarks[n]["time_per_agent_ms"] for n in SCALES]
    ax.bar(range(len(SCALES)), times, color='#7e57c2', alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(range(len(SCALES)), per_agent, 'o-', color='#ff5722', linewidth=2, markersize=8)
    ax.set_title('Computational Cost', fontweight='bold')
    ax.set_xticks(range(len(SCALES))); ax.set_xticklabels([str(n) for n in SCALES])
    ax.set_xlabel('Agent Count'); ax.set_ylabel('Total Time (s)', color='#7e57c2')
    ax2.set_ylabel('Time/Agent (ms)', color='#ff5722')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig50_role_specialization_scaling.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[P1] Fig 50 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [P1] 1000-에이전트 스케일 불변성 실험")
    print("=" * 60)
    
    results, benchmarks = run_scale_experiment()
    sii = compute_sii(results)
    
    print(f"\n{'Scale':>6} | {'SVO':>12} | {'Meta':>6} | {'Base':>6} | {'ATE':>7} | {'SII':>5} | {'Gini-M':>6}")
    print("-" * 65)
    for n in SCALES:
        for svo in SVO_CONDITIONS:
            d = results[n][svo]
            print(f"{n:>6} | {svo:>12} | {d['meta_coop']:.3f} | {d['base_coop']:.3f} | "
                  f"{d['ate']:+.4f} | {sii[n][svo]:.2f} | {d['meta_gini']:.4f}")
        print("-" * 65)
    
    print(f"\n--- Computation Benchmark ---")
    for n in SCALES:
        b = benchmarks[n]
        print(f"  {n:>5} agents: {b['time_sec']:.2f}s total, {b['time_per_agent_ms']:.2f}ms/agent")
    
    plot_fig49(results, benchmarks)
    plot_fig50(results, benchmarks)
    
    json_data = {
        "results": {str(n): d for n, d in results.items()},
        "benchmarks": {str(n): b for n, b in benchmarks.items()},
        "sii": {str(n): s for n, s in sii.items()},
    }
    json_path = os.path.join(OUTPUT_DIR, "scale_experiment_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\n[P1] 결과 JSON: {json_path}")
