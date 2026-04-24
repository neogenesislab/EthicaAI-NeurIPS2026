"""
EthicaAI M3: 커뮤니케이션 채널 실험 (Cheap Talk)
Phase M — 에이전트 간 의도 신호가 협력에 미치는 영향

실험 설계: 2×2 조건 (Meta ON/OFF) × (Communication ON/OFF)
- Communication ON: 에이전트가 행동 전 '의도' 메시지 전송
- 메시지 진실성(truthfulness) 자체가 학습 대상
- Fig 27: Communication 효과 비교
- Fig 28: 메시지 진실성 학습 곡선
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
N_STEPS = 300

SVO_CONDITIONS = {
    'selfish': 0, 'prosocial': 45, 'cooperative': 60, 'altruistic': 75,
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


def sim_pgg_with_communication(theta_deg, use_meta, use_comm, seed):
    """PGG with optional communication channel."""
    rng = np.random.RandomState(seed)
    
    E = 20.0
    M = 1.6
    N = N_AGENTS
    
    wealth = np.zeros(N)
    # 에이전트별 진실성 성향 (0=항상 거짓, 1=항상 진실)
    truthfulness = np.ones(N) * 0.5  # 초기 50% 진실
    learning_rate = 0.02
    
    coop_hist, reward_hist, truth_hist = [], [], []
    comm_effect_hist = []
    
    for t in range(N_STEPS):
        resource_est = np.mean(wealth + E) / (2 * E)
        
        # === Phase 1: Communication (메시지 전송) ===
        messages = np.zeros(N)  # 0=defect 신호, 1=cooperate 신호
        actual_intents = np.zeros(N)
        
        for i in range(N):
            lam = compute_lambda(theta_deg, min(1.0, max(0, resource_est)), use_meta)
            actual_intents[i] = lam  # 실제 기여 의도
            
            if use_comm:
                # 진실한 메시지? 또는 거짓 메시지?
                if rng.random() < truthfulness[i]:
                    messages[i] = 1.0 if lam > 0.5 else 0.0  # 진실
                else:
                    messages[i] = 0.0 if lam > 0.5 else 1.0  # 거짓
        
        # === Phase 2: 메시지 관찰 후 행동 결정 ===
        contributions = np.zeros(N)
        for i in range(N):
            lam = compute_lambda(theta_deg, min(1.0, max(0, resource_est)), use_meta)
            
            # Communication 효과: 타인의 메시지를 관찰하여 기여도 조정
            if use_comm:
                others_msg = np.delete(messages, i)
                avg_signal = np.mean(others_msg)
                
                # 타인이 협력 신호 → 나도 더 기여 (reciprocity)
                comm_boost = 0.15 * (avg_signal - 0.5)
                lam = max(0, min(1.0, lam + comm_boost))
            
            c = lam * E + rng.normal(0, E * 0.05)
            contributions[i] = max(0, min(E, c))
        
        # === Phase 3: 결과 계산 ===
        total_c = np.sum(contributions)
        public_good = total_c * M / N
        
        for i in range(N):
            payoff = (E - contributions[i]) + public_good
            wealth[i] += payoff - E
        
        # === Phase 4: 진실성 학습 ===
        if use_comm:
            for i in range(N):
                # 진실한 메시지를 보낸 에이전트가 더 높은 보상을 받으면 진실성 증가
                was_truthful = (messages[i] == (1.0 if actual_intents[i] > 0.5 else 0.0))
                payoff_i = (E - contributions[i]) + public_good
                avg_payoff = np.mean([(E - contributions[j]) + public_good for j in range(N)])
                
                if was_truthful and payoff_i >= avg_payoff:
                    truthfulness[i] = min(1.0, truthfulness[i] + learning_rate)
                elif not was_truthful and payoff_i < avg_payoff:
                    truthfulness[i] = min(1.0, truthfulness[i] + learning_rate * 0.5)
                else:
                    truthfulness[i] = max(0.0, truthfulness[i] - learning_rate * 0.3)
        
        coop_hist.append(float(total_c / (N * E)))
        reward_hist.append(float(np.mean([(E - contributions[j]) + public_good for j in range(N)])))
        truth_hist.append(float(np.mean(truthfulness)) if use_comm else 0.5)
    
    return {
        'cooperation': float(np.mean(coop_hist[-80:])),
        'reward': float(np.mean(reward_hist[-80:])),
        'truthfulness': float(np.mean(truth_hist[-80:])),
        'coop_trajectory': coop_hist,
        'truth_trajectory': truth_hist,
    }


def run_experiment(output_dir):
    """2×2 × 4 SVO 실험."""
    results = {}
    conditions = [
        ('meta_comm', True, True),
        ('meta_only', True, False),
        ('comm_only', False, True),
        ('baseline', False, False),
    ]
    
    for svo_name, theta in SVO_CONDITIONS.items():
        results[svo_name] = {}
        
        for cond_name, use_meta, use_comm in conditions:
            runs = [sim_pgg_with_communication(theta, use_meta, use_comm, s) for s in range(N_SEEDS)]
            
            results[svo_name][cond_name] = {
                'cooperation': float(np.mean([r['cooperation'] for r in runs])),
                'cooperation_std': float(np.std([r['cooperation'] for r in runs])),
                'reward': float(np.mean([r['reward'] for r in runs])),
                'truthfulness': float(np.mean([r['truthfulness'] for r in runs])),
                'coop_trajectories': [r['coop_trajectory'] for r in runs],
                'truth_trajectories': [r['truth_trajectory'] for r in runs],
            }
        
        print(f"  ✓ {svo_name} 완료")
    
    return results


def plot_communication(results, output_dir):
    """Fig 27: Communication 효과 + Fig 28: 진실성 학습."""
    svos = list(results.keys())
    conditions = ['meta_comm', 'meta_only', 'comm_only', 'baseline']
    cond_labels = ['Meta+Comm', 'Meta Only', 'Comm Only', 'Baseline']
    cond_colors = ['#4fc3f7', '#66bb6a', '#ffa726', '#888']
    
    # ── Fig 27: Communication 효과 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Fig 27. Communication Channel Effect on Cooperation', 
                 fontsize=14, fontweight='bold')
    
    # (A) 조건별 협력률
    ax = axes[0]
    x = np.arange(len(svos))
    width = 0.2
    for idx, (cond, label, color) in enumerate(zip(conditions, cond_labels, cond_colors)):
        vals = [results[svo][cond]['cooperation'] for svo in svos]
        errs = [results[svo][cond]['cooperation_std'] for svo in svos]
        ax.bar(x + idx * width, vals, width, yerr=errs, label=label, 
               color=color, alpha=0.85, capsize=2)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"{SVO_CONDITIONS[s]}°" for s in svos], fontsize=9)
    ax.set_ylabel('Cooperation Rate')
    ax.set_xlabel('SVO Angle')
    ax.legend(fontsize=8, ncol=2)
    ax.set_title('(A) Cooperation by Condition')
    ax.set_ylim(0, 1.05)
    
    # (B) 협력 궤적 (prosocial 예시)
    ax = axes[1]
    for cond, label, color in zip(conditions, cond_labels, cond_colors):
        trajs = results['prosocial'][cond]['coop_trajectories']
        mean_traj = np.mean(trajs, axis=0)
        ax.plot(mean_traj, color=color, lw=2, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cooperation Rate')
    ax.legend(fontsize=9)
    ax.set_title('(B) Cooperation Trajectory (θ=45°)')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    out27 = os.path.join(output_dir, 'fig27_communication.png')
    plt.savefig(out27, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[M3] Fig 27 저장: {out27}")
    
    # ── Fig 28: 진실성 학습 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Fig 28. Message Truthfulness Learning', fontsize=14, fontweight='bold')
    
    # (A) SVO별 진실성 수렴
    ax = axes[0]
    for svo_name in svos:
        trajs = results[svo_name]['meta_comm']['truth_trajectories']
        mean_traj = np.mean(trajs, axis=0)
        ax.plot(mean_traj, lw=2, label=f"{SVO_CONDITIONS[svo_name]}° {svo_name}")
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Truthfulness')
    ax.legend(fontsize=9)
    ax.set_title('(A) Truthfulness Over Time (Meta+Comm)')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='#888', ls=':', alpha=0.5, label='Initial')
    
    # (B) 최종 진실성 vs SVO
    ax = axes[1]
    for cond, label, color in zip(['meta_comm', 'comm_only'], ['Meta+Comm', 'Comm Only'], ['#4fc3f7', '#ffa726']):
        vals = [results[svo][cond]['truthfulness'] for svo in svos]
        ax.bar([SVO_CONDITIONS[s] for s in svos], vals, 
               width=8, label=label, color=color, alpha=0.8)
    ax.set_xlabel('SVO Angle')
    ax.set_ylabel('Final Truthfulness')
    ax.legend(fontsize=9)
    ax.set_title('(B) Final Truthfulness by SVO')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    out28 = os.path.join(output_dir, 'fig28_truthfulness.png')
    plt.savefig(out28, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[M3] Fig 28 저장: {out28}")
    
    return out27, out28


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "simulation/outputs/reproduce"
    os.makedirs(output_dir, exist_ok=True)
    
    print("[M3] 커뮤니케이션 채널 실험 시작...")
    results = run_experiment(output_dir)
    plot_communication(results, output_dir)
    
    print("\n" + "=" * 60)
    print("M3 COMMUNICATION SUMMARY")
    print("=" * 60)
    for svo in results:
        mc = results[svo]['meta_comm']['cooperation']
        mo = results[svo]['meta_only']['cooperation']
        boost = mc - mo
        truth = results[svo]['meta_comm']['truthfulness']
        print(f"  {svo:>12} | Meta+Comm: {mc:.3f} | Meta Only: {mo:.3f} | "
              f"Comm boost: {boost:+.3f} | Truth: {truth:.3f}")
    
    save = {svo: {cond: {k: v for k, v in d.items() if 'trajectories' not in k}
                  for cond, d in svo_data.items()}
            for svo, svo_data in results.items()}
    json_path = os.path.join(output_dir, 'communication_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save, f, indent=2, ensure_ascii=False)
    print(f"\n[M3] 결과 JSON: {json_path}")
