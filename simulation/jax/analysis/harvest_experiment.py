"""
EthicaAI L1: Harvest 환경 교차 검증
Phase L — 4번째 환경에서 Meta-Ranking 효과 확인

핵심 역학:
- 매 스텝 고정량의 자원이 재생 (carrying capacity 제한)
- 협력자는 적게 수확하고, 자원 보전에도 기여
- 비협력자는 많이 수확하지만 자원 보전 안 함
- Meta-Ranking: 자원 위기 시 절제 강화, 풍요 시 SVO 기본 행동
"""
import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SVO_CONDITIONS = {
    'selfish': 0, 'individualist': 15, 'competitive': 30,
    'prosocial': 45, 'cooperative': 60, 'altruistic': 75, 'full_altruist': 90,
}
N_SEEDS = 10
N_AGENTS = 20
N_STEPS = 300


def simulate_harvest(theta_deg, use_meta, seed):
    """Harvest 시뮬레이션 — 상수 재성장 + 협력자 보전 보너스."""
    rng = np.random.RandomState(seed)
    
    capacity = 100.0
    resource = 80.0
    base_regrowth = 3.0  # 매 스텝 고정 재생량
    
    wealth = np.zeros(N_AGENTS)
    res_hist, coop_hist = [], []
    
    for t in range(N_STEPS):
        theta_rad = np.radians(theta_deg)
        base_coop = np.sin(theta_rad)
        
        cooperators = 0
        agent_harvests = []
        
        for i in range(N_AGENTS):
            # Meta-Ranking 동적 조절
            if use_meta:
                ratio = resource / capacity
                if ratio < 0.3:
                    coop_prob = min(1.0, base_coop + 0.5)  # 위기: 대폭 절제
                elif ratio > 0.7:
                    coop_prob = base_coop  # 풍요: 기본 행동
                else:
                    coop_prob = min(1.0, base_coop + 0.2 * (1.0 - ratio))  # 점진 조절
            else:
                coop_prob = base_coop
            
            if rng.random() < coop_prob:
                agent_harvests.append(0.3)  # 절제 수확
                cooperators += 1
            else:
                agent_harvests.append(1.2)  # 과다 수확
        
        # 수확
        total_demand = sum(agent_harvests)
        scale = min(1.0, resource / max(total_demand, 1e-6))
        actual = [h * scale for h in agent_harvests]
        resource -= sum(actual)
        
        # 재성장: 고정량 + 협력자 보너스
        coop_bonus = cooperators * 0.2  # 협력자가 보전 활동
        regrowth = base_regrowth + coop_bonus
        resource = min(capacity, resource + regrowth)
        resource = max(0.0, resource)
        
        # 부 업데이트
        for i in range(N_AGENTS):
            wealth[i] += actual[i]
        
        res_hist.append(resource / capacity)
        coop_hist.append(cooperators / N_AGENTS)
    
    return {
        'sustainability': float(np.mean(res_hist[-80:])),
        'cooperation': float(np.mean(coop_hist[-80:])),
        'mean_wealth': float(np.mean(wealth)),
        'resource_trajectory': res_hist,
        'coop_trajectory': coop_hist,
    }


def run_experiment(output_dir):
    """전체 실험."""
    results = {}
    for svo_name, theta in SVO_CONDITIONS.items():
        meta_runs = [simulate_harvest(theta, True, s) for s in range(N_SEEDS)]
        base_runs = [simulate_harvest(theta, False, s) for s in range(N_SEEDS)]
        
        results[svo_name] = {
            'theta': theta,
            'meta': {
                'sustainability': float(np.mean([r['sustainability'] for r in meta_runs])),
                'sustainability_std': float(np.std([r['sustainability'] for r in meta_runs])),
                'cooperation': float(np.mean([r['cooperation'] for r in meta_runs])),
                'wealth': float(np.mean([r['mean_wealth'] for r in meta_runs])),
            },
            'baseline': {
                'sustainability': float(np.mean([r['sustainability'] for r in base_runs])),
                'sustainability_std': float(np.std([r['sustainability'] for r in base_runs])),
                'cooperation': float(np.mean([r['cooperation'] for r in base_runs])),
                'wealth': float(np.mean([r['mean_wealth'] for r in base_runs])),
            },
            'meta_trajectories': [r['resource_trajectory'] for r in meta_runs],
            'base_trajectories': [r['resource_trajectory'] for r in base_runs],
        }
    return results


def plot_harvest(results, output_dir):
    """Figure 20: Harvest 교차 검증."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Fig 20. Harvest Environment Cross-Validation', fontsize=14, fontweight='bold')
    
    svos = list(results.keys())
    thetas = [results[s]['theta'] for s in svos]
    x = np.arange(len(svos))
    w = 0.35
    
    # (A) 자원 지속성
    ax = axes[0]
    m_s = [results[s]['meta']['sustainability'] for s in svos]
    b_s = [results[s]['baseline']['sustainability'] for s in svos]
    m_e = [results[s]['meta']['sustainability_std'] for s in svos]
    b_e = [results[s]['baseline']['sustainability_std'] for s in svos]
    ax.bar(x - w/2, m_s, w, yerr=m_e, label='Meta-Ranking', color='#4fc3f7', alpha=0.85, capsize=3)
    ax.bar(x + w/2, b_s, w, yerr=b_e, label='Baseline (Static SVO)', color='#888', alpha=0.65, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}°" for t in thetas], fontsize=8)
    ax.set_ylabel('Resource Sustainability')
    ax.set_xlabel('SVO Angle')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title('(A) Sustainability by SVO')
    ax.set_ylim(0, 1.0)
    
    # (B) 협력률 비교
    ax = axes[1]
    m_c = [results[s]['meta']['cooperation'] for s in svos]
    b_c = [results[s]['baseline']['cooperation'] for s in svos]
    ax.plot(thetas, m_c, 'o-', color='#4fc3f7', label='Meta-Ranking', lw=2, ms=6)
    ax.plot(thetas, b_c, 's--', color='#888', label='Baseline', lw=2, ms=6)
    ax.fill_between(thetas, m_c, b_c, alpha=0.12, color='#4fc3f7')
    ax.set_xlabel('SVO Angle (°)')
    ax.set_ylabel('Cooperation Rate')
    ax.legend(fontsize=9)
    ax.set_title('(B) Cooperation Rates')
    ax.set_ylim(0, 1.05)
    
    # (C) 궤적 (selfish vs prosocial)
    ax = axes[2]
    colors = {'selfish': '#ff5252', 'prosocial': '#4fc3f7'}
    for svo_key in ['selfish', 'prosocial']:
        d = results[svo_key]
        meta_mean = np.mean(d['meta_trajectories'], axis=0)
        base_mean = np.mean(d['base_trajectories'], axis=0)
        ax.plot(meta_mean, color=colors[svo_key], lw=2, label=f'{svo_key} Meta')
        ax.plot(base_mean, color=colors[svo_key], lw=1.5, ls='--', alpha=0.6, label=f'{svo_key} Base')
    ax.set_xlabel('Step')
    ax.set_ylabel('Resource Level')
    ax.legend(fontsize=8, ncol=2, loc='lower left')
    ax.set_title('(C) Resource Trajectories')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    out = os.path.join(output_dir, 'fig20_harvest.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[L1] Figure 저장: {out}")
    return out


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "simulation/outputs/reproduce"
    os.makedirs(output_dir, exist_ok=True)
    
    print("[L1] Harvest 환경 교차 검증 시작...")
    results = run_experiment(output_dir)
    plot_harvest(results, output_dir)
    
    print("\n--- HARVEST RESULTS ---")
    for svo, r in results.items():
        d = r['meta']['sustainability'] - r['baseline']['sustainability']
        print(f"  {svo:15s} | Meta: {r['meta']['sustainability']:.3f} | "
              f"Base: {r['baseline']['sustainability']:.3f} | Δ: {d:+.3f}")
    
    # 통계 요약
    all_deltas = [r['meta']['sustainability'] - r['baseline']['sustainability'] for r in results.values()]
    print(f"\n  평균 Δ(sustainability): {np.mean(all_deltas):+.3f}")
    print(f"  최대 Δ: {max(all_deltas):+.3f} ({list(results.keys())[np.argmax(all_deltas)]})")
    
    save_data = {k: {kk: vv for kk, vv in v.items() if 'trajectories' not in kk} 
                 for k, v in results.items()}
    json_path = os.path.join(output_dir, 'harvest_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"[L1] 결과 JSON: {json_path}")
