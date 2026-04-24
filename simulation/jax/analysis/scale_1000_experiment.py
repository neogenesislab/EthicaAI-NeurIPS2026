"""
EthicaAI L2: 1000-에이전트 확장성 실험
Phase L — 에이전트 수 증가에 따른 Meta-Ranking 효과 크기 추이 분석

대규모(20→100→500→1000) 시뮬레이션을 통해
메타-랭킹 효과의 초선형 증가 여부를 검증합니다.
"""
import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# 설정
SCALES = [20, 50, 100, 200, 500, 1000]
SVO_THETA = 45  # Prosocial 기준
N_SEEDS = 10
N_STEPS = 150


def compute_lambda(theta_deg, wealth, use_meta=True):
    """동적 λ 계산."""
    theta = np.radians(theta_deg)
    lb = np.sin(theta)
    if not use_meta:
        return lb
    if wealth < 2.0:
        return 0.0
    elif wealth > 8.0:
        return min(1.0, 1.5 * lb)
    return lb


def simulate_scale(n_agents, theta_deg, use_meta, seed):
    """N-에이전트 공유자원 시뮬레이션."""
    rng = np.random.RandomState(seed)
    
    max_resource = n_agents * 5.0
    resource = max_resource
    regrowth_rate = 0.15
    
    wealth = np.zeros(n_agents)
    rewards = []
    
    for t in range(N_STEPS):
        lambdas = np.array([compute_lambda(theta_deg, wealth[i], use_meta) for i in range(n_agents)])
        
        # 수확 결정: λ 높을수록 적게 수확
        harvest_rate = 1.0 - lambdas * 0.5
        harvest_rate += rng.normal(0, 0.03, n_agents)
        harvest_rate = np.clip(harvest_rate, 0.1, 1.0)
        
        demand = harvest_rate * 0.5
        total_demand = demand.sum()
        
        # 재성장 먼저
        resource += regrowth_rate * resource * (1.0 - resource / max_resource) + n_agents * 0.1
        resource = max(0, min(max_resource, resource))
        
        if total_demand > resource:
            actual = resource * (demand / total_demand)
        else:
            actual = demand
        
        wealth += actual
        resource -= actual.sum()
        resource = max(0, resource)
        
        rewards.append(np.mean(actual))
    
    # 사회적 지표
    sorted_w = np.sort(wealth)
    n = len(sorted_w)
    gini = (2 * np.sum((np.arange(1, n+1) - 0.5) * sorted_w) / (n * np.sum(sorted_w))) - 1 if np.sum(sorted_w) > 0 else 0
    
    return {
        'mean_reward': float(np.mean(rewards[-30:])),
        'gini': float(max(0, gini)),
        'final_resource': float(resource / max_resource),
    }


def cohens_f_squared(meta_vals, base_vals):
    """Cohen's f² 효과 크기 계산."""
    all_vals = meta_vals + base_vals
    grand_mean = np.mean(all_vals)
    ss_total = np.sum((np.array(all_vals) - grand_mean)**2)
    ss_between = len(meta_vals) * (np.mean(meta_vals) - grand_mean)**2 + \
                 len(base_vals) * (np.mean(base_vals) - grand_mean)**2
    ss_within = ss_total - ss_between
    if ss_within <= 0:
        return float('inf')
    return float(ss_between / ss_within)


def run_scale_experiment(output_dir):
    """모든 스케일에서 실험 실행."""
    results = {}
    
    for n in SCALES:
        print(f"  Scale N={n}...")
        meta_runs = [simulate_scale(n, SVO_THETA, True, s) for s in range(N_SEEDS)]
        base_runs = [simulate_scale(n, SVO_THETA, False, s) for s in range(N_SEEDS)]
        
        m_rewards = [r['mean_reward'] for r in meta_runs]
        b_rewards = [r['mean_reward'] for r in base_runs]
        m_gini = [r['gini'] for r in meta_runs]
        b_gini = [r['gini'] for r in base_runs]
        
        f2_reward = cohens_f_squared(m_rewards, b_rewards)
        f2_gini = cohens_f_squared(m_gini, b_gini)
        
        results[n] = {
            'n_agents': n,
            'f2_reward': f2_reward,
            'f2_gini': f2_gini,
            'meta_reward': float(np.mean(m_rewards)),
            'base_reward': float(np.mean(b_rewards)),
            'meta_gini': float(np.mean(m_gini)),
            'base_gini': float(np.mean(b_gini)),
            'meta_resource': float(np.mean([r['final_resource'] for r in meta_runs])),
            'base_resource': float(np.mean([r['final_resource'] for r in base_runs])),
        }
    
    return results


def plot_scale(results, output_dir):
    """Figure 21: 확장성 분석."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Fig 21. Scalability: Meta-Ranking Effect by Agent Count", fontsize=14, fontweight='bold')
    
    scales = sorted(results.keys())
    
    # 1. Cohen's f² 추이
    ax = axes[0]
    f2r = [results[n]['f2_reward'] for n in scales]
    f2g = [results[n]['f2_gini'] for n in scales]
    ax.plot(scales, f2r, 'o-', color='#4fc3f7', label="f² (Reward)", linewidth=2, markersize=6)
    ax.plot(scales, f2g, 's-', color='#ce93d8', label="f² (Gini)", linewidth=2, markersize=6)
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel("Cohen's f²")
    ax.set_xscale('log')
    ax.legend()
    ax.set_title('(A) Effect Size Scaling')
    ax.axhline(y=0.35, color='gray', linestyle='--', alpha=0.5, label='Large effect threshold')
    
    # 2. 자원 지속 가능성
    ax = axes[1]
    mr = [results[n]['meta_resource'] for n in scales]
    br = [results[n]['base_resource'] for n in scales]
    ax.plot(scales, mr, 'o-', color='#66bb6a', label='Meta-Ranking', linewidth=2)
    ax.plot(scales, br, 's--', color='#888', label='Baseline', linewidth=2)
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Final Resource (normalized)')
    ax.set_xscale('log')
    ax.legend()
    ax.set_title('(B) Resource Sustainability')
    
    # 3. Gini 계수
    ax = axes[2]
    mg = [results[n]['meta_gini'] for n in scales]
    bg = [results[n]['base_gini'] for n in scales]
    ax.bar([str(n) for n in scales], bg, 0.35, label='Baseline', color='#888', alpha=0.6)
    ax.bar([str(n) for n in scales], mg, 0.35, label='Meta-Ranking', color='#4fc3f7', alpha=0.8,
           bottom=[0]*len(scales))
    # 오버레이 라인
    ax.plot(range(len(scales)), mg, 'o-', color='#4fc3f7', linewidth=2)
    ax.plot(range(len(scales)), bg, 's--', color='#666', linewidth=2)
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Gini Coefficient (↓ better)')
    ax.legend()
    ax.set_title('(C) Inequality')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fig21_scale_1000.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[L2] Figure 저장: {out_path}")
    return out_path


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "simulation/outputs/reproduce"
    os.makedirs(output_dir, exist_ok=True)
    
    print("[L2] 1000-에이전트 확장성 실험 시작...")
    results = run_scale_experiment(output_dir)
    plot_scale(results, output_dir)
    
    print("\n--- SCALE RESULTS ---")
    for n in sorted(results.keys()):
        r = results[n]
        print(f"  N={n:5d} | f²(R)={r['f2_reward']:.3f} | f²(G)={r['f2_gini']:.3f} | "
              f"Meta R={r['meta_reward']:.4f} | Base R={r['base_reward']:.4f}")
    
    json_path = os.path.join(output_dir, 'scale_1000_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"[L2] 결과 JSON: {json_path}")
