"""
P6: 적대적 에이전트 강건성 테스트
EthicaAI Phase P — NeurIPS 방법론 보강

메타랭킹의 Byzantine Robustness 검증:
- 적대적 에이전트(Free-rider, Exploiter, Random, Sybil) 비율 변화에 따른 성능 변화
- 메타랭킹 인구의 회복 탄력성(Resilience) 측정

출력: Fig 55 (강건성 곡선), Fig 56 (회복 탄력성)
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

# === 설정 ===
N_AGENTS = 50
N_STEPS = 300
N_SEEDS = 10
ENDOWMENT = 100
ADV_FRACTIONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


class ProsocialAgent:
    """메타랭킹 에이전트 (θ=45°)"""
    def __init__(self, rng):
        self.lambda_t = np.sin(np.radians(45.0))
        self.rng = rng
    
    def decide(self, resource, group_avg):
        base = np.sin(np.radians(45.0))
        if resource < 0.2:
            target = base * 0.3
        elif resource > 0.7:
            target = min(1, base * 1.5)
        else:
            reciprocity = group_avg / (ENDOWMENT + 1e-8)
            target = base * (0.7 + 0.6 * reciprocity)
        self.lambda_t = 0.9 * self.lambda_t + 0.1 * target
        return int(np.clip(self.lambda_t * ENDOWMENT * 0.8 + self.rng.normal(0, 3), 0, ENDOWMENT))


class FreeRider:
    """무임승차자: 절대 기여하지 않음"""
    def __init__(self, rng):
        self.lambda_t = 0.0
        self.rng = rng
    def decide(self, resource, group_avg):
        return int(np.clip(self.rng.normal(2, 1), 0, ENDOWMENT))


class Exploiter:
    """착취자: 처음에 협력했다가 배신"""
    def __init__(self, rng):
        self.lambda_t = 0.7
        self.rng = rng
        self.t = 0
    def decide(self, resource, group_avg):
        self.t += 1
        if self.t < 30:
            self.lambda_t = 0.7
        else:
            self.lambda_t = max(0, self.lambda_t - 0.03)
        return int(np.clip(self.lambda_t * ENDOWMENT * 0.8 + self.rng.normal(0, 3), 0, ENDOWMENT))


class RandomAgent:
    """무작위 행동"""
    def __init__(self, rng):
        self.lambda_t = 0.5
        self.rng = rng
    def decide(self, resource, group_avg):
        self.lambda_t = self.rng.random()
        return self.rng.randint(0, ENDOWMENT + 1)


class SybilAttacker:
    """시빌 공격자: 협력 신호를 주면서 실제로는 적게 기여"""
    def __init__(self, rng):
        self.lambda_t = 0.8  # 높은 λ 신호 (기만)
        self.rng = rng
    def decide(self, resource, group_avg):
        self.lambda_t = 0.8  # 신호는 높게
        actual_contrib = 10  # 실제 기여는 낮게
        return int(np.clip(actual_contrib + self.rng.normal(0, 2), 0, ENDOWMENT))


ADV_TYPES = {
    "free_rider": FreeRider,
    "exploiter": Exploiter,
    "random": RandomAgent,
    "sybil": SybilAttacker,
}


def simulate_adversarial(adv_type, adv_fraction, seed):
    """적대적 에이전트 혼합 시뮬레이션"""
    rng = np.random.RandomState(seed)
    
    n_adv = int(N_AGENTS * adv_fraction)
    n_pro = N_AGENTS - n_adv
    
    agents = [ProsocialAgent(rng) for _ in range(n_pro)] + \
             [ADV_TYPES[adv_type](rng) for _ in range(n_adv)]
    
    resource = 0.5
    coop_history = []
    welfare_history = []
    resource_history = []
    
    for t in range(N_STEPS):
        group_avg = np.mean([a.lambda_t for a in agents]) * ENDOWMENT * 0.8
        contributions = [a.decide(resource, group_avg) for a in agents]
        
        total = sum(contributions)
        public_good = total * 1.6 / N_AGENTS
        payoffs = [(ENDOWMENT - c) + public_good for c in contributions]
        
        resource = np.clip(resource + 0.02 * (np.mean(contributions) / ENDOWMENT - 0.3), 0, 1)
        
        # 메타랭킹 에이전트만의 협력률
        pro_contribs = contributions[:n_pro]
        if pro_contribs:
            coop = np.mean([c > ENDOWMENT * 0.3 for c in pro_contribs])
        else:
            coop = 0.0
        
        coop_history.append(coop)
        welfare_history.append(np.mean(payoffs))
        resource_history.append(resource)
    
    return {
        "coop_history": coop_history,
        "welfare_history": welfare_history,
        "resource_history": resource_history,
        "mean_coop": float(np.mean(coop_history[-30:])),
        "mean_welfare": float(np.mean(welfare_history[-30:])),
        "sustainability": float(np.mean([r > 0.1 for r in resource_history])),
        "recovery_time": _recovery_time(resource_history),
    }


def _recovery_time(resource_history):
    """위기 후 회복 시간 (자원 < 0.2 후 0.5 복귀까지)"""
    in_crisis = False
    crisis_start = None
    recoveries = []
    
    for t, r in enumerate(resource_history):
        if r < 0.2 and not in_crisis:
            in_crisis = True
            crisis_start = t
        elif r > 0.5 and in_crisis:
            in_crisis = False
            if crisis_start is not None:
                recoveries.append(t - crisis_start)
    
    return float(np.mean(recoveries)) if recoveries else float('inf')


def run_robustness_experiment():
    """전체 강건성 실험"""
    results = {}
    for adv_type in ADV_TYPES:
        results[adv_type] = {}
        for frac in ADV_FRACTIONS:
            runs = [simulate_adversarial(adv_type, frac, s) for s in range(N_SEEDS)]
            results[adv_type][frac] = {
                "mean_coop": float(np.mean([r["mean_coop"] for r in runs])),
                "mean_welfare": float(np.mean([r["mean_welfare"] for r in runs])),
                "sustainability": float(np.mean([r["sustainability"] for r in runs])),
                "recovery_time": float(np.mean([r["recovery_time"] for r in runs if r["recovery_time"] != float('inf')])) if any(r["recovery_time"] != float('inf') for r in runs) else float('inf'),
            }
    return results


def plot_fig55(results):
    """Fig 55: 강건성 곡선 — 적대적 비율 vs 성능"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Fig 55: Byzantine Robustness — Meta-Ranking Under Adversarial Agents",
                 fontsize=14, fontweight='bold', y=1.02)
    
    adv_colors = {'free_rider': '#e53935', 'exploiter': '#ff9800', 'random': '#7e57c2', 'sybil': '#00897b'}
    fracs_pct = [f * 100 for f in ADV_FRACTIONS]
    
    # 좌: 협력률
    ax = axes[0]
    for adv_type, color in adv_colors.items():
        vals = [results[adv_type][f]["mean_coop"] for f in ADV_FRACTIONS]
        ax.plot(fracs_pct, vals, 'o-', color=color, linewidth=2, markersize=6, label=adv_type.replace('_', ' ').title())
    ax.set_title('Cooperation Rate vs Adversary Fraction', fontweight='bold')
    ax.set_xlabel('Adversary Fraction (%)'); ax.set_ylabel('Cooperation Rate')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # 중: 복지
    ax = axes[1]
    for adv_type, color in adv_colors.items():
        vals = [results[adv_type][f]["mean_welfare"] for f in ADV_FRACTIONS]
        ax.plot(fracs_pct, vals, 's-', color=color, linewidth=2, markersize=6, label=adv_type.replace('_', ' ').title())
    ax.set_title('Social Welfare vs Adversary Fraction', fontweight='bold')
    ax.set_xlabel('Adversary Fraction (%)'); ax.set_ylabel('Welfare')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # 우: 지속가능성
    ax = axes[2]
    for adv_type, color in adv_colors.items():
        vals = [results[adv_type][f]["sustainability"] * 100 for f in ADV_FRACTIONS]
        ax.plot(fracs_pct, vals, '^-', color=color, linewidth=2, markersize=6, label=adv_type.replace('_', ' ').title())
    ax.axhline(80, color='green', linestyle=':', alpha=0.5, label='80% Threshold')
    ax.set_title('Sustainability vs Adversary Fraction', fontweight='bold')
    ax.set_xlabel('Adversary Fraction (%)'); ax.set_ylabel('Sustainability (%)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig55_adversarial_robustness.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[P6] Fig 55 저장: {path}")


def plot_fig56(results):
    """Fig 56: 회복 탄력성 히트맵"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle("Fig 56: Resilience Analysis — Recovery from Adversarial Shocks",
                 fontsize=14, fontweight='bold', y=1.02)
    
    adv_types = list(ADV_TYPES.keys())
    
    # 좌: 감내 한계 (Tolerance Threshold) — 성능 50% 유지 가능한 최대 적대 비율
    ax = axes[0]
    for adv_type in adv_types:
        baseline = results[adv_type][0.0]["mean_coop"]
        threshold = 0.0
        for frac in ADV_FRACTIONS:
            if results[adv_type][frac]["mean_coop"] >= baseline * 0.5:
                threshold = frac
        ax.barh(adv_type.replace('_', ' ').title(), threshold * 100,
               color={'free_rider': '#e53935', 'exploiter': '#ff9800', 'random': '#7e57c2', 'sybil': '#00897b'}[adv_type],
               alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.text(threshold * 100 + 1, adv_type.replace('_', ' ').title(), f'{threshold*100:.0f}%',
               va='center', fontweight='bold')
    ax.set_title('Tolerance Threshold (50% Performance Retained)', fontweight='bold')
    ax.set_xlabel('Max Adversary Fraction (%)')
    ax.grid(True, axis='x', alpha=0.3)
    
    # 우: 성능 저하 히트맵
    ax = axes[1]
    degradation = np.zeros((len(adv_types), len(ADV_FRACTIONS)))
    for i, adv_type in enumerate(adv_types):
        baseline = results[adv_type][0.0]["mean_coop"]
        for j, frac in enumerate(ADV_FRACTIONS):
            if baseline > 0:
                degradation[i, j] = results[adv_type][frac]["mean_coop"] / baseline * 100
            else:
                degradation[i, j] = 100
    
    im = ax.imshow(degradation, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(ADV_FRACTIONS)))
    ax.set_xticklabels([f'{f*100:.0f}%' for f in ADV_FRACTIONS], fontsize=8)
    ax.set_yticks(range(len(adv_types)))
    ax.set_yticklabels([t.replace('_', ' ').title() for t in adv_types])
    ax.set_title('Performance Retention (%)', fontweight='bold')
    ax.set_xlabel('Adversary Fraction')
    
    for i in range(len(adv_types)):
        for j in range(len(ADV_FRACTIONS)):
            text = f'{degradation[i, j]:.0f}'
            color = 'white' if degradation[i, j] < 50 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontweight='bold', fontsize=9, color=color)
    
    plt.colorbar(im, ax=ax, label='% of Baseline Performance')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig56_resilience.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[P6] Fig 56 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [P6] 적대적 에이전트 강건성 테스트")
    print("=" * 60)
    
    results = run_robustness_experiment()
    
    print(f"\n{'Type':>12} | {'Frac':>5} | {'Coop':>6} | {'Welfare':>8} | {'Sustain':>7}")
    print("-" * 50)
    for adv_type in ADV_TYPES:
        for frac in [0.0, 0.1, 0.2, 0.3, 0.5]:
            if frac in results[adv_type]:
                d = results[adv_type][frac]
                print(f"{adv_type:>12} | {frac*100:4.0f}% | {d['mean_coop']:.3f} | {d['mean_welfare']:8.1f} | {d['sustainability']*100:5.1f}%")
        print("-" * 50)
    
    plot_fig55(results)
    plot_fig56(results)
    
    json_data = {at: {str(f): d for f, d in fd.items()} for at, fd in results.items()}
    json_path = os.path.join(OUTPUT_DIR, "adversarial_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\n[P6] 결과 JSON: {json_path}")
