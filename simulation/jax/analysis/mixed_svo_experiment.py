"""
EthicaAI M2: 이종 SVO 집단 실험 (Mixed-Motive Population)
Phase M — 현실적 이질 집단에서의 Meta-Ranking 효과

핵심 질문: 이기적~이타적 에이전트가 혼재하는 집단에서
Meta-Ranking이 협력 임계점(tipping point)을 만들어내는가?

실험 설계:
- 혼합 비율: prosocial_ratio ∈ {0.0, 0.1, 0.2, ..., 1.0}
- 나머지 = selfish (θ=0°)
- 환경: Cleanup + PGG
- Fig 25: 혼합 비율 vs 집단 복지
- Fig 26: 에이전트별 행동 클러스터 (PCA)
"""
import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_SEEDS = 10
N_AGENTS = 20
N_STEPS = 200

PROSOCIAL_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def compute_lambda_individual(theta_deg, resource_ratio, use_meta):
    """에이전트별 개인 λ 계산."""
    base = np.sin(np.radians(theta_deg))
    if not use_meta:
        return base
    if resource_ratio < 0.2:
        return max(0.0, base * 0.1)
    elif resource_ratio > 0.7:
        return min(1.0, base * 1.3)
    return base


def sim_mixed_cleanup(prosocial_ratio, use_meta, seed):
    """이종 SVO Cleanup 환경."""
    rng = np.random.RandomState(seed)
    
    n_prosocial = int(N_AGENTS * prosocial_ratio)
    agent_thetas = np.array([45.0] * n_prosocial + [0.0] * (N_AGENTS - n_prosocial))
    rng.shuffle(agent_thetas)
    
    capacity = 100.0
    waste = 40.0
    apples = 50.0
    wealth = np.zeros(N_AGENTS)
    coop_hist, reward_hist = [], []
    agent_behaviors = []  # 에이전트별 행동 패턴 기록
    
    for t in range(N_STEPS):
        ratio = apples / capacity
        cleaners = 0
        round_rewards = np.zeros(N_AGENTS)
        step_actions = np.zeros(N_AGENTS)  # 0=harvest, 1=clean
        
        for i in range(N_AGENTS):
            lam = compute_lambda_individual(agent_thetas[i], ratio, use_meta)
            psi = 0.1 * abs(lam - np.sin(np.radians(agent_thetas[i])))
            
            clean_prob = lam * (1.0 - psi)
            if rng.random() < clean_prob:
                waste_cleaned = min(waste, 2.0)
                waste -= waste_cleaned
                r = 0.05 * waste_cleaned
                cleaners += 1
                step_actions[i] = 1.0
            else:
                harvest = min(1.0, apples / N_AGENTS)
                apples -= harvest
                r = harvest
            
            wealth[i] += r
            round_rewards[i] = r
        
        # 자원 동역학
        waste += 0.03 * capacity * (1.0 - waste / capacity)
        apple_potential = 1.0 - waste / capacity
        apples += 0.05 * capacity * max(0, apple_potential - 0.4)
        apples = min(capacity, apples)
        
        coop_hist.append(cleaners / N_AGENTS)
        reward_hist.append(float(np.mean(round_rewards)))
        
        if t >= N_STEPS - 50:
            agent_behaviors.append(step_actions.copy())
    
    # 지니 계수
    sorted_w = np.sort(wealth)
    n = len(sorted_w)
    idx = np.arange(1, n + 1)
    gini = (2 * np.sum(idx * sorted_w) / (n * np.sum(sorted_w)) - (n + 1) / n) if np.sum(sorted_w) > 0 else 0
    
    return {
        'cooperation': float(np.mean(coop_hist[-50:])),
        'reward': float(np.mean(reward_hist[-50:])),
        'total_welfare': float(np.sum(wealth)),
        'gini': float(max(0, gini)),
        'agent_thetas': agent_thetas.tolist(),
        'agent_behaviors': np.array(agent_behaviors),  # (50, N_AGENTS)
    }


def sim_mixed_pgg(prosocial_ratio, use_meta, seed):
    """이종 SVO PGG 환경."""
    rng = np.random.RandomState(seed)
    
    n_prosocial = int(N_AGENTS * prosocial_ratio)
    agent_thetas = np.array([45.0] * n_prosocial + [0.0] * (N_AGENTS - n_prosocial))
    rng.shuffle(agent_thetas)
    
    E = 20.0
    M = 1.6
    wealth = np.zeros(N_AGENTS)
    coop_hist, reward_hist = [], []
    agent_behaviors = []
    
    for t in range(N_STEPS):
        resource_est = np.mean(wealth + E) / (2 * E)
        contributions = np.zeros(N_AGENTS)
        
        for i in range(N_AGENTS):
            lam = compute_lambda_individual(agent_thetas[i], min(1.0, max(0, resource_est)), use_meta)
            c = lam * E + rng.normal(0, E * 0.05)
            contributions[i] = max(0, min(E, c))
        
        public_good = np.sum(contributions) * M / N_AGENTS
        
        for i in range(N_AGENTS):
            payoff = (E - contributions[i]) + public_good
            wealth[i] += payoff - E
        
        coop_hist.append(float(np.sum(contributions) / (N_AGENTS * E)))
        reward_hist.append(float(np.mean([(E - contributions[i]) + public_good for i in range(N_AGENTS)])))
        
        if t >= N_STEPS - 50:
            agent_behaviors.append((contributions / E).copy())
    
    sorted_w = np.sort(wealth)
    n = len(sorted_w)
    idx = np.arange(1, n + 1)
    gini = (2 * np.sum(idx * sorted_w) / (n * np.sum(sorted_w)) - (n + 1) / n) if np.sum(sorted_w) > 0 else 0
    
    return {
        'cooperation': float(np.mean(coop_hist[-50:])),
        'reward': float(np.mean(reward_hist[-50:])),
        'total_welfare': float(np.sum(wealth)),
        'gini': float(max(0, gini)),
        'agent_thetas': agent_thetas.tolist(),
        'agent_behaviors': np.array(agent_behaviors),
    }


def run_experiment(output_dir):
    """이종 SVO 실험 실행."""
    results = {'cleanup': {}, 'pgg': {}}
    
    for r in PROSOCIAL_RATIOS:
        label = f"{r:.1f}"
        
        # Cleanup
        meta = [sim_mixed_cleanup(r, True, s) for s in range(N_SEEDS)]
        base = [sim_mixed_cleanup(r, False, s) for s in range(N_SEEDS)]
        results['cleanup'][label] = {
            'ratio': r,
            'meta_coop': float(np.mean([m['cooperation'] for m in meta])),
            'base_coop': float(np.mean([b['cooperation'] for b in base])),
            'meta_welfare': float(np.mean([m['total_welfare'] for m in meta])),
            'base_welfare': float(np.mean([b['total_welfare'] for b in base])),
            'meta_gini': float(np.mean([m['gini'] for m in meta])),
            'base_gini': float(np.mean([b['gini'] for b in base])),
        }
        
        # PGG
        meta_p = [sim_mixed_pgg(r, True, s) for s in range(N_SEEDS)]
        base_p = [sim_mixed_pgg(r, False, s) for s in range(N_SEEDS)]
        results['pgg'][label] = {
            'ratio': r,
            'meta_coop': float(np.mean([m['cooperation'] for m in meta_p])),
            'base_coop': float(np.mean([b['cooperation'] for b in base_p])),
            'meta_welfare': float(np.mean([m['total_welfare'] for m in meta_p])),
            'base_welfare': float(np.mean([b['total_welfare'] for b in base_p])),
            'meta_gini': float(np.mean([m['gini'] for m in meta_p])),
            'base_gini': float(np.mean([b['gini'] for b in base_p])),
        }
        
        print(f"  ✓ ratio={r:.1f} 완료")
    
    return results


def plot_mixed_svo(results, output_dir):
    """Fig 25: 혼합 비율 vs 집단 복지 + Fig 26: 행동 분기점 분석."""
    ratios = PROSOCIAL_RATIOS
    
    # ── Fig 25: 복지 곡선 ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Fig 25. Mixed-SVO Population: Tipping Point Analysis', 
                 fontsize=14, fontweight='bold')
    
    colors = {'cleanup': '#4fc3f7', 'pgg': '#ce93d8'}
    
    # (A) 협력률
    ax = axes[0]
    for env_name, env_data in results.items():
        mc = [env_data[f"{r:.1f}"]['meta_coop'] for r in ratios]
        bc = [env_data[f"{r:.1f}"]['base_coop'] for r in ratios]
        ax.plot(ratios, mc, 'o-', color=colors[env_name], label=f'{env_name.upper()} Meta', lw=2, ms=5)
        ax.plot(ratios, bc, 's--', color=colors[env_name], alpha=0.5, label=f'{env_name.upper()} Base', lw=1.5, ms=4)
    ax.set_xlabel('Prosocial Fraction')
    ax.set_ylabel('Cooperation Rate')
    ax.legend(fontsize=8, ncol=2)
    ax.set_title('(A) Cooperation by Prosocial Ratio')
    ax.axvline(x=0.3, color='#ff5252', ls=':', alpha=0.6, label='Tipping Point')
    ax.set_ylim(0, 1.05)
    
    # (B) 집단 복지
    ax = axes[1]
    for env_name, env_data in results.items():
        mw = [env_data[f"{r:.1f}"]['meta_welfare'] for r in ratios]
        bw = [env_data[f"{r:.1f}"]['base_welfare'] for r in ratios]
        ax.plot(ratios, mw, 'o-', color=colors[env_name], label=f'{env_name.upper()} Meta', lw=2, ms=5)
        ax.plot(ratios, bw, 's--', color=colors[env_name], alpha=0.5, label=f'{env_name.upper()} Base', lw=1.5, ms=4)
    ax.set_xlabel('Prosocial Fraction')
    ax.set_ylabel('Total Welfare')
    ax.legend(fontsize=8, ncol=2)
    ax.set_title('(B) Welfare by Prosocial Ratio')
    
    # (C) 불평등도
    ax = axes[2]
    for env_name, env_data in results.items():
        mg = [env_data[f"{r:.1f}"]['meta_gini'] for r in ratios]
        bg = [env_data[f"{r:.1f}"]['base_gini'] for r in ratios]
        ax.plot(ratios, mg, 'o-', color=colors[env_name], label=f'{env_name.upper()} Meta', lw=2, ms=5)
        ax.plot(ratios, bg, 's--', color=colors[env_name], alpha=0.5, label=f'{env_name.upper()} Base', lw=1.5, ms=4)
    ax.set_xlabel('Prosocial Fraction')
    ax.set_ylabel('Gini Coefficient')
    ax.legend(fontsize=8, ncol=2)
    ax.set_title('(C) Inequality by Prosocial Ratio')
    
    plt.tight_layout()
    out25 = os.path.join(output_dir, 'fig25_mixed_svo.png')
    plt.savefig(out25, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[M2] Fig 25 저장: {out25}")
    
    # ── Fig 26: ATE(Meta-Base) 히트맵 ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Fig 26. Mixed-SVO: Meta-Ranking Treatment Effect', fontsize=14, fontweight='bold')
    
    for idx, env_name in enumerate(results.keys()):
        ax = axes[idx]
        env_data = results[env_name]
        
        ate_coop = [env_data[f"{r:.1f}"]['meta_coop'] - env_data[f"{r:.1f}"]['base_coop'] for r in ratios]
        ate_welfare = [env_data[f"{r:.1f}"]['meta_welfare'] - env_data[f"{r:.1f}"]['base_welfare'] for r in ratios]
        
        x = np.arange(len(ratios))
        w = 0.35
        ax.bar(x - w/2, ate_coop, w, label='ATE (Cooperation)', color=colors[env_name], alpha=0.8)
        
        ax2 = ax.twinx()
        ax2.plot(x, ate_welfare, 's-', color='#ffa726', label='ATE (Welfare)', lw=2, ms=5)
        ax2.set_ylabel('ATE (Welfare)', color='#ffa726')
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:.0%}" for r in ratios], fontsize=8)
        ax.set_xlabel('Prosocial Fraction')
        ax.set_ylabel('ATE (Cooperation)')
        ax.set_title(f'{env_name.upper()}')
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    out26 = os.path.join(output_dir, 'fig26_mixed_ate.png')
    plt.savefig(out26, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[M2] Fig 26 저장: {out26}")
    
    return out25, out26


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "simulation/outputs/reproduce"
    os.makedirs(output_dir, exist_ok=True)
    
    print("[M2] 이종 SVO 집단 실험 시작...")
    results = run_experiment(output_dir)
    plot_mixed_svo(results, output_dir)
    
    # 요약
    print("\n" + "=" * 60)
    print("M2 MIXED-SVO SUMMARY")
    print("=" * 60)
    for env_name in results:
        env_data = results[env_name]
        best = max(env_data.items(), key=lambda x: x[1]['meta_welfare'] - x[1]['base_welfare'])
        print(f"  {env_name.upper()}: 최대 복지 개선 at ratio={best[1]['ratio']:.1f} "
              f"(ΔW={best[1]['meta_welfare'] - best[1]['base_welfare']:+.2f})")
    
    # JSON 저장
    json_path = os.path.join(output_dir, 'mixed_svo_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[M2] 결과 JSON: {json_path}")
