"""
EthicaAI M1: 4환경 × 7 SVO Full Sweep 실험
Phase M — 전체 실험 공간 커버
환경: Cleanup, IPD, PGG, Harvest
SVO: selfish(0°) ~ full_altruist(90°)
시드: 10개 (통계적 안정성)

각 환경에 적합한 시뮬레이션 로직으로 Meta-Ranking 효과를 측정.
ATE (Average Treatment Effect) = Meta ON - Meta OFF
"""
import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product

# ── 상수 정의 (환경별 설정은 ENV_CONFIGS 참조) ──
SVO_CONDITIONS = {
    'selfish': 0, 'individualist': 15, 'competitive': 30,
    'prosocial': 45, 'cooperative': 60, 'altruistic': 75, 'full_altruist': 90,
}

N_SEEDS = 10
N_AGENTS = 20
N_STEPS = 200

ENV_CONFIGS = {
    'cleanup': {
        'capacity': 100.0,
        'waste_spawn': 0.03,
        'apple_respawn': 0.05,
        'threshold_depletion': 0.4,
    },
    'ipd': {
        'T': 5.0, 'R': 3.0, 'P': 1.0, 'S': 0.0,
    },
    'pgg': {
        'endowment': 20.0,
        'multiplier': 1.6,
        'n_players': 4,
    },
    'harvest': {
        'capacity': 100.0,
        'base_regrowth': 3.0,
    },
}


def compute_lambda(theta_deg, resource_ratio, use_meta):
    """동적 λ 계산 (Meta-Ranking의 핵심 메커니즘)."""
    base = np.sin(np.radians(theta_deg))
    if not use_meta:
        return base
    
    # 자원 상태 기반 동적 조절
    if resource_ratio < 0.2:
        return max(0.0, base * 0.1)  # 생존 위기 → 자기보전
    elif resource_ratio > 0.7:
        return min(1.0, base * 1.3)  # 풍요 → 헌신 강화
    else:
        return base


# ═══════════════════════════════════════════
# 환경별 시뮬레이션 함수들
# ═══════════════════════════════════════════

def sim_cleanup(theta_deg, use_meta, seed):
    """Cleanup 환경: 폐기물 청소 vs 사과 수확 딜레마."""
    rng = np.random.RandomState(seed)
    cfg = ENV_CONFIGS['cleanup']
    
    waste = cfg['threshold_depletion'] * cfg['capacity']
    apples = cfg['capacity'] * 0.5
    wealth = np.zeros(N_AGENTS)
    coop_hist, reward_hist, gini_hist = [], [], []
    
    for t in range(N_STEPS):
        resource_ratio = apples / cfg['capacity']
        cleaners = 0
        round_rewards = []
        
        for i in range(N_AGENTS):
            lam = compute_lambda(theta_deg, resource_ratio, use_meta)
            psi = 0.1 * abs(lam - np.sin(np.radians(theta_deg)))  # 자기통제 비용
            
            # 협력 = 청소, 비협력 = 수확
            clean_prob = lam * (1.0 - psi)
            if rng.random() < clean_prob:
                # 청소: 폐기물 제거, 보상 낮음
                waste_cleaned = min(waste, 2.0)
                waste -= waste_cleaned
                r = cfg['apple_respawn'] * waste_cleaned  # 간접 보상
                cleaners += 1
            else:
                # 수확: 사과 획득
                harvest = min(1.0, apples / N_AGENTS)
                apples -= harvest
                r = harvest
            
            wealth[i] += r
            round_rewards.append(r)
        
        # 자원 동역학
        waste += cfg['waste_spawn'] * cfg['capacity'] * (1.0 - waste / cfg['capacity'])
        apple_potential = 1.0 - waste / cfg['capacity']
        apples += cfg['apple_respawn'] * cfg['capacity'] * max(0, apple_potential - cfg['threshold_depletion'])
        apples = min(cfg['capacity'], apples)
        
        coop_hist.append(cleaners / N_AGENTS)
        reward_hist.append(np.mean(round_rewards))
        
        # 지니 계수
        sorted_w = np.sort(wealth)
        n = len(sorted_w)
        idx = np.arange(1, n + 1)
        gini = (2 * np.sum(idx * sorted_w) / (n * np.sum(sorted_w)) - (n + 1) / n) if np.sum(sorted_w) > 0 else 0
        gini_hist.append(max(0, gini))
    
    return {
        'cooperation': float(np.mean(coop_hist[-50:])),
        'reward': float(np.mean(reward_hist[-50:])),
        'gini': float(np.mean(gini_hist[-50:])),
    }


def sim_ipd(theta_deg, use_meta, seed):
    """IPD 환경: 2인 반복 죄수 딜레마."""
    rng = np.random.RandomState(seed)
    cfg = ENV_CONFIGS['ipd']
    
    # 10쌍 × 200라운드
    n_pairs = 10
    all_coop, all_reward = [], []
    
    for pair in range(n_pairs):
        wealth = [0.0, 0.0]
        coop_counts = [0, 0]
        round_rewards = []
        
        for t in range(N_STEPS):
            actions = []
            for i in range(2):
                j = 1 - i
                total = coop_counts[j] + (t - coop_counts[j])
                pcr = coop_counts[j] / max(total, 1)
                resource_est = (wealth[i] + 10) / 20.0  # 정규화
                
                lam = compute_lambda(theta_deg, min(1.0, resource_est), use_meta)
                
                # 기대 보상 비교
                ev_coop = pcr * cfg['R'] + (1 - pcr) * cfg['S']
                ev_defect = pcr * cfg['T'] + (1 - pcr) * cfg['P']
                ev_meta = (1 - lam) * ev_defect + lam * ev_coop
                
                coop_prob = 1 / (1 + np.exp(-(ev_meta - ev_defect) * 2))
                actions.append(int(rng.random() > coop_prob))  # 0=C, 1=D
            
            payoff_mat = [[cfg['R'], cfg['S']], [cfg['T'], cfg['P']]]
            for i in range(2):
                r = payoff_mat[actions[i]][actions[1 - i]]
                wealth[i] += r
                if actions[i] == 0:
                    coop_counts[i] += 1
            
            round_rewards.append(sum(payoff_mat[actions[0]][actions[1]] for _ in range(1)) / 2)
            all_coop.append(int(actions[0] == 0))
            all_coop.append(int(actions[1] == 0))
        
        all_reward.extend([np.mean(wealth) / N_STEPS])
    
    return {
        'cooperation': float(np.mean(all_coop[-100:])),
        'reward': float(np.mean(all_reward)),
        'gini': 0.0,  # 2인 게임은 대칭
    }


def sim_pgg(theta_deg, use_meta, seed):
    """PGG 환경: 공공재 게임."""
    rng = np.random.RandomState(seed)
    cfg = ENV_CONFIGS['pgg']
    n = cfg['n_players']
    E = cfg['endowment']
    M = cfg['multiplier']
    
    wealth = np.zeros(n)
    coop_hist, reward_hist = [], []
    
    for t in range(N_STEPS):
        resource_est = np.mean(wealth + E) / (2 * E)
        contributions = []
        
        for i in range(n):
            lam = compute_lambda(theta_deg, min(1.0, max(0, resource_est)), use_meta)
            c = lam * E + rng.normal(0, E * 0.05)
            c = max(0, min(E, c))
            contributions.append(c)
        
        total_c = sum(contributions)
        public_good = total_c * M / n
        
        for i in range(n):
            payoff = (E - contributions[i]) + public_good
            wealth[i] += payoff - E  # 순이익
        
        coop_hist.append(total_c / (n * E))
        reward_hist.append(np.mean([E - contributions[i] + public_good for i in range(n)]))
    
    # 지니
    sorted_w = np.sort(wealth)
    nn = len(sorted_w)
    idx = np.arange(1, nn + 1)
    gini = (2 * np.sum(idx * sorted_w) / (nn * np.sum(sorted_w)) - (nn + 1) / nn) if np.sum(sorted_w) > 0 else 0
    
    return {
        'cooperation': float(np.mean(coop_hist[-50:])),
        'reward': float(np.mean(reward_hist[-50:])),
        'gini': float(max(0, gini)),
    }


def sim_harvest(theta_deg, use_meta, seed):
    """Harvest 환경: 공유 자원 수확."""
    rng = np.random.RandomState(seed)
    cfg = ENV_CONFIGS['harvest']
    
    resource = cfg['capacity'] * 0.8
    wealth = np.zeros(N_AGENTS)
    coop_hist, reward_hist = [], []
    
    for t in range(N_STEPS):
        ratio = resource / cfg['capacity']
        cooperators = 0
        round_harvest = []
        
        for i in range(N_AGENTS):
            lam = compute_lambda(theta_deg, ratio, use_meta)
            
            coop_prob = lam
            if use_meta and ratio < 0.3:
                coop_prob = min(1.0, lam + 0.5)
            
            if rng.random() < coop_prob:
                h = 0.3  # 절제
                cooperators += 1
            else:
                h = 1.2  # 과다
            round_harvest.append(h)
        
        total_demand = sum(round_harvest)
        scale = min(1.0, resource / max(total_demand, 1e-6))
        actual = [h * scale for h in round_harvest]
        resource -= sum(actual)
        
        coop_bonus = cooperators * 0.2
        resource = min(cfg['capacity'], resource + cfg['base_regrowth'] + coop_bonus)
        resource = max(0, resource)
        
        for i in range(N_AGENTS):
            wealth[i] += actual[i]
        
        coop_hist.append(cooperators / N_AGENTS)
        reward_hist.append(np.mean(actual))
    
    sorted_w = np.sort(wealth)
    nn = len(sorted_w)
    idx = np.arange(1, nn + 1)
    gini = (2 * np.sum(idx * sorted_w) / (nn * np.sum(sorted_w)) - (nn + 1) / nn) if np.sum(sorted_w) > 0 else 0
    
    return {
        'cooperation': float(np.mean(coop_hist[-50:])),
        'reward': float(np.mean(reward_hist[-50:])),
        'gini': float(max(0, gini)),
    }


# ═══════════════════════════════════════════
# Full Sweep 메인 로직
# ═══════════════════════════════════════════

SIM_FUNCS = {
    'cleanup': sim_cleanup,
    'ipd': sim_ipd,
    'pgg': sim_pgg,
    'harvest': sim_harvest,
}


def run_full_sweep(output_dir):
    """4환경 × 7 SVO × 10 seeds 전체 실험."""
    results = {}
    total = len(SIM_FUNCS) * len(SVO_CONDITIONS) * N_SEEDS * 2
    done = 0
    
    for env_name, sim_fn in SIM_FUNCS.items():
        results[env_name] = {}
        
        for svo_name, theta in SVO_CONDITIONS.items():
            meta_runs = []
            base_runs = []
            
            for seed in range(N_SEEDS):
                meta_runs.append(sim_fn(theta, True, seed))
                base_runs.append(sim_fn(theta, False, seed))
                done += 2
            
            # ATE 계산 (Meta - Baseline)
            ate_coop = np.mean([r['cooperation'] for r in meta_runs]) - np.mean([r['cooperation'] for r in base_runs])
            ate_reward = np.mean([r['reward'] for r in meta_runs]) - np.mean([r['reward'] for r in base_runs])
            ate_gini = np.mean([r['gini'] for r in meta_runs]) - np.mean([r['gini'] for r in base_runs])
            
            results[env_name][svo_name] = {
                'theta': theta,
                'meta_coop': float(np.mean([r['cooperation'] for r in meta_runs])),
                'base_coop': float(np.mean([r['cooperation'] for r in base_runs])),
                'ate_coop': float(ate_coop),
                'meta_reward': float(np.mean([r['reward'] for r in meta_runs])),
                'base_reward': float(np.mean([r['reward'] for r in base_runs])),
                'ate_reward': float(ate_reward),
                'meta_gini': float(np.mean([r['gini'] for r in meta_runs])),
                'base_gini': float(np.mean([r['gini'] for r in base_runs])),
                'ate_gini': float(ate_gini),
            }
        
        print(f"  ✓ {env_name} 완료 ({done}/{total})")
    
    return results


def plot_full_sweep(results, output_dir):
    """Fig 24: 4환경 × 7 SVO Full Sweep 히트맵."""
    envs = list(results.keys())
    svos = list(SVO_CONDITIONS.keys())
    thetas = list(SVO_CONDITIONS.values())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Fig 24. Full Sweep: 4 Environments × 7 SVO Conditions', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    metrics = [
        ('ate_coop', 'ATE (Cooperation Rate)', 'RdBu'),
        ('ate_reward', 'ATE (Mean Reward)', 'RdBu'),
        ('ate_gini', 'ATE (Gini Coefficient)', 'RdBu_r'),
    ]
    
    for ax_idx, (metric_key, title, cmap) in enumerate(metrics):
        ax = axes[ax_idx]
        
        # 히트맵 데이터 구성
        data = np.zeros((len(envs), len(svos)))
        for i, env in enumerate(envs):
            for j, svo in enumerate(svos):
                data[i, j] = results[env][svo][metric_key]
        
        # 대칭 범위
        vmax = max(abs(data.min()), abs(data.max()))
        vmin = -vmax if metric_key != 'ate_gini' else -vmax
        
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)
        
        # 레이블
        ax.set_xticks(range(len(svos)))
        ax.set_xticklabels([f"{t}°" for t in thetas], fontsize=8, rotation=45)
        ax.set_yticks(range(len(envs)))
        ax.set_yticklabels([e.upper() for e in envs], fontsize=9)
        ax.set_xlabel('SVO Angle')
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        # 셀 값 텍스트
        for i in range(len(envs)):
            for j in range(len(svos)):
                val = data[i, j]
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{val:+.3f}', ha='center', va='center', 
                       fontsize=7, color=color, fontweight='bold')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fig24_full_sweep.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[M1] Figure 저장: {out_path}")
    return out_path


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "simulation/outputs/reproduce"
    os.makedirs(output_dir, exist_ok=True)
    
    print("[M1] 4환경 × 7 SVO Full Sweep 시작...")
    results = run_full_sweep(output_dir)
    plot_full_sweep(results, output_dir)
    
    # 요약 통계
    print("\n" + "=" * 70)
    print("M1 FULL SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'환경':>10} | {'최대 ATE(Coop)':>16} | {'최대 ATE(Reward)':>16} | {'최적 SVO':>10}")
    print("-" * 70)
    for env_name, env_data in results.items():
        best_svo = max(env_data.items(), key=lambda x: x[1]['ate_coop'])
        best_r_svo = max(env_data.items(), key=lambda x: x[1]['ate_reward'])
        print(f"{env_name:>10} | {best_svo[1]['ate_coop']:>+16.4f} | "
              f"{best_r_svo[1]['ate_reward']:>+16.4f} | {best_svo[0]:>10}")
    
    # JSON 저장
    json_path = os.path.join(output_dir, 'full_sweep_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[M1] 결과 JSON: {json_path}")
