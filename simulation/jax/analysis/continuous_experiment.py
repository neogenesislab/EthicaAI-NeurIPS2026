"""
EthicaAI M4: 연속 행동 공간 PGG 실험
Phase M — 이산 행동 → 연속 기여로 확장

핵심: 기여도가 {0, 1} 이산이 아닌 [0.0, 1.0] 연속인 환경에서
Meta-Ranking이 어떤 기여 분포를 만들어내는가?

Fig 29: 이산 vs 연속 기여 분포 비교
Fig 30: 연속 환경에서의 λ_t 동적 궤적
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

SVO_CONDITIONS = {
    'selfish': 0, 'individualist': 15, 'prosocial': 45,
    'cooperative': 60, 'altruistic': 75,
}


def compute_lambda(theta_deg, resource_ratio, use_meta):
    """동적 λ 계산."""
    base = np.sin(np.radians(theta_deg))
    if not use_meta:
        return base
    if resource_ratio < 0.2:
        return max(0.0, base * 0.1)
    elif resource_ratio > 0.7:
        return min(1.0, base * 1.3)
    return base


def sim_continuous_pgg(theta_deg, use_meta, seed, continuous=True):
    """연속/이산 PGG 시뮬레이션."""
    rng = np.random.RandomState(seed)
    
    E = 20.0
    M = 1.6
    N = N_AGENTS
    
    wealth = np.zeros(N)
    coop_hist, reward_hist = [], []
    lambda_hist = []
    contribution_dist = []  # 마지막 50스텝의 기여 분포
    
    for t in range(N_STEPS):
        resource_est = np.mean(wealth + E) / (2 * E)
        contributions = np.zeros(N)
        lambdas = np.zeros(N)
        
        for i in range(N):
            lam = compute_lambda(theta_deg, min(1.0, max(0, resource_est)), use_meta)
            lambdas[i] = lam
            
            if continuous:
                # 연속: Beta 분포 기반 기여 (mode ≈ λ)
                alpha_param = max(0.5, lam * 5)
                beta_param = max(0.5, (1 - lam) * 5)
                c_ratio = rng.beta(alpha_param, beta_param)
                contributions[i] = c_ratio * E
            else:
                # 이산: 확률적 이진 선택
                if rng.random() < lam:
                    contributions[i] = E  # 전액 기여
                else:
                    contributions[i] = 0  # 기여 안 함
        
        total_c = np.sum(contributions)
        public_good = total_c * M / N
        
        for i in range(N):
            payoff = (E - contributions[i]) + public_good
            wealth[i] += payoff - E
        
        coop_hist.append(float(total_c / (N * E)))
        reward_hist.append(float(np.mean([(E - contributions[j]) + public_good for j in range(N)])))
        lambda_hist.append(float(np.mean(lambdas)))
        
        if t >= N_STEPS - 50:
            contribution_dist.extend((contributions / E).tolist())
    
    return {
        'cooperation': float(np.mean(coop_hist[-50:])),
        'reward': float(np.mean(reward_hist[-50:])),
        'lambda_trajectory': lambda_hist,
        'contribution_dist': contribution_dist,
        'coop_trajectory': coop_hist,
    }


def run_experiment(output_dir):
    """연속 vs 이산 비교 실험."""
    results = {}
    
    for svo_name, theta in SVO_CONDITIONS.items():
        cont_meta = [sim_continuous_pgg(theta, True, s, True) for s in range(N_SEEDS)]
        cont_base = [sim_continuous_pgg(theta, False, s, True) for s in range(N_SEEDS)]
        disc_meta = [sim_continuous_pgg(theta, True, s, False) for s in range(N_SEEDS)]
        disc_base = [sim_continuous_pgg(theta, False, s, False) for s in range(N_SEEDS)]
        
        results[svo_name] = {
            'theta': theta,
            'cont_meta': {
                'cooperation': float(np.mean([r['cooperation'] for r in cont_meta])),
                'reward': float(np.mean([r['reward'] for r in cont_meta])),
                'contribution_dist': [d for r in cont_meta for d in r['contribution_dist']],
                'lambda_trajectories': [r['lambda_trajectory'] for r in cont_meta],
                'coop_trajectories': [r['coop_trajectory'] for r in cont_meta],
            },
            'cont_base': {
                'cooperation': float(np.mean([r['cooperation'] for r in cont_base])),
                'reward': float(np.mean([r['reward'] for r in cont_base])),
                'contribution_dist': [d for r in cont_base for d in r['contribution_dist']],
            },
            'disc_meta': {
                'cooperation': float(np.mean([r['cooperation'] for r in disc_meta])),
                'reward': float(np.mean([r['reward'] for r in disc_meta])),
                'contribution_dist': [d for r in disc_meta for d in r['contribution_dist']],
            },
            'disc_base': {
                'cooperation': float(np.mean([r['cooperation'] for r in disc_base])),
                'reward': float(np.mean([r['reward'] for r in disc_base])),
                'contribution_dist': [d for r in disc_base for d in r['contribution_dist']],
            },
        }
        print(f"  ✓ {svo_name} 완료")
    
    return results


def plot_continuous(results, output_dir):
    """Fig 29: 기여 분포 + Fig 30: λ 궤적."""
    svos = list(results.keys())
    
    # ── Fig 29: 이산 vs 연속 기여 분포 ──
    fig, axes = plt.subplots(2, len(svos), figsize=(3.5 * len(svos), 8))
    fig.suptitle('Fig 29. Discrete vs Continuous Contribution Distributions', 
                 fontsize=14, fontweight='bold')
    
    for j, svo in enumerate(svos):
        d = results[svo]
        theta = d['theta']
        
        # 이산 (위)
        ax = axes[0, j]
        disc_m = d['disc_meta']['contribution_dist']
        disc_b = d['disc_base']['contribution_dist']
        bins_d = np.linspace(-0.05, 1.05, 12)
        ax.hist(disc_m, bins=bins_d, alpha=0.7, color='#4fc3f7', label='Meta', density=True)
        ax.hist(disc_b, bins=bins_d, alpha=0.5, color='#888', label='Base', density=True)
        ax.set_title(f'Discrete (θ={theta}°)', fontsize=10)
        ax.set_xlim(-0.1, 1.1)
        if j == 0:
            ax.set_ylabel('Density')
        ax.legend(fontsize=7)
        
        # 연속 (아래)
        ax = axes[1, j]
        cont_m = d['cont_meta']['contribution_dist']
        cont_b = d['cont_base']['contribution_dist']
        bins_c = np.linspace(0, 1, 30)
        ax.hist(cont_m, bins=bins_c, alpha=0.7, color='#4fc3f7', label='Meta', density=True)
        ax.hist(cont_b, bins=bins_c, alpha=0.5, color='#888', label='Base', density=True)
        ax.set_title(f'Continuous (θ={theta}°)', fontsize=10)
        ax.set_xlabel('Contribution / Endowment')
        ax.set_xlim(0, 1)
        if j == 0:
            ax.set_ylabel('Density')
        ax.legend(fontsize=7)
    
    plt.tight_layout()
    out29 = os.path.join(output_dir, 'fig29_continuous_dist.png')
    plt.savefig(out29, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[M4] Fig 29 저장: {out29}")
    
    # ── Fig 30: λ 동적 궤적 + 협력 비교 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Fig 30. Continuous PGG: Lambda Dynamics & Cooperation', 
                 fontsize=14, fontweight='bold')
    
    colors = ['#ff5252', '#ffa726', '#4fc3f7', '#66bb6a', '#ce93d8']
    
    # (A) λ 궤적
    ax = axes[0]
    for i, svo in enumerate(svos):
        d = results[svo]
        trajs = d['cont_meta']['lambda_trajectories']
        mean_traj = np.mean(trajs, axis=0)
        ax.plot(mean_traj, color=colors[i], lw=2, label=f'{svo} ({d["theta"]}°)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean λ_t')
    ax.legend(fontsize=8)
    ax.set_title('(A) Dynamic λ Trajectories (Continuous PGG)')
    ax.set_ylim(0, 1.05)
    
    # (B) 이산 vs 연속 협력률 비교
    ax = axes[1]
    x = np.arange(len(svos))
    w = 0.2
    bars = [
        ('disc_meta', 'Discrete Meta', '#4fc3f7'),
        ('disc_base', 'Discrete Base', '#888'),
        ('cont_meta', 'Contin. Meta', '#66bb6a'),
        ('cont_base', 'Contin. Base', '#bbb'),
    ]
    for idx, (key, label, color) in enumerate(bars):
        vals = [results[svo][key]['cooperation'] for svo in svos]
        ax.bar(x + idx * w, vals, w, label=label, color=color, alpha=0.85)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels([f"{results[s]['theta']}°" for s in svos], fontsize=9)
    ax.set_ylabel('Cooperation Rate')
    ax.legend(fontsize=7, ncol=2)
    ax.set_title('(B) Discrete vs Continuous Cooperation')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    out30 = os.path.join(output_dir, 'fig30_continuous_lambda.png')
    plt.savefig(out30, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[M4] Fig 30 저장: {out30}")
    
    return out29, out30


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "simulation/outputs/reproduce"
    os.makedirs(output_dir, exist_ok=True)
    
    print("[M4] 연속 행동 공간 PGG 실험 시작...")
    results = run_experiment(output_dir)
    plot_continuous(results, output_dir)
    
    print("\n" + "=" * 60)
    print("M4 CONTINUOUS ACTION SPACE SUMMARY")
    print("=" * 60)
    for svo in results:
        d = results[svo]
        cm = d['cont_meta']['cooperation']
        cb = d['cont_base']['cooperation']
        dm = d['disc_meta']['cooperation']
        db = d['disc_base']['cooperation']
        print(f"  {svo:>12} | Cont Meta: {cm:.3f} | Cont Base: {cb:.3f} | "
              f"Disc Meta: {dm:.3f} | Disc Base: {db:.3f}")
    
    save = {svo: {k: {kk: vv for kk, vv in v.items() if 'dist' not in kk and 'traj' not in kk}
                  for k, v in data.items() if isinstance(v, dict)}
            for svo, data in results.items()}
    json_path = os.path.join(output_dir, 'continuous_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[M4] 결과 JSON: {json_path}")
