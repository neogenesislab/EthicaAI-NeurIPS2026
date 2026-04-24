"""
O8: 파일럿 실험 분석 + 시뮬레이션 데모
EthicaAI Phase O — 연구축 IV: Human-in-the-Loop

oTree 실험 데이터 분석 스크립트.
실제 데이터가 없을 때는 시뮬레이션 데이터로 Fig 47~48 생성.

출력: Fig 47 (인간-AI 협력 패턴), Fig 48 (인간 적응 패턴)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import os
import sys

# AIPartner 임포트 (상대 경로)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from experiments.otree_pgg.pgg_experiment import AIPartner
except ImportError:
    # 직접 정의 (폴백)
    class AIPartner:
        SVO_CONFIGS = {
            'selfish': {'theta': 0.0, 'noise': 0.05},
            'prosocial': {'theta': 45.0, 'noise': 0.05},
            'meta_ranking': {'theta': 45.0, 'noise': 0.03},
        }
        def __init__(self, condition='meta_ranking'):
            self.condition = condition
            config = self.SVO_CONFIGS.get(condition, self.SVO_CONFIGS['meta_ranking'])
            self.svo_theta = np.radians(config['theta'])
            self.noise = config['noise']
            self.lambda_t = np.sin(self.svo_theta)
        def decide(self, round_number, avg=50, resource=0.5, endowment=100):
            base = np.sin(self.svo_theta) * endowment * 0.8
            if self.condition == 'meta_ranking':
                if resource < 0.2: base *= 0.3
                elif resource > 0.7: base *= 1.5
            elif self.condition == 'selfish': base = 0
            return int(np.clip(base + np.random.normal(0, self.noise * endowment), 0, endowment))


OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "ETHICAAI_OUTPUT_DIR", "simulation/outputs/reproduce"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 설정 ===
N_ROUNDS = 20
N_SIMULATED_HUMANS = 10
ENDOWMENT = 100
MULTIPLIER = 1.8
AI_CONDITIONS = ['selfish', 'prosocial', 'meta_ranking']


def simulate_human(condition_ai, seed):
    """인간 행동 시뮬레이션 (다양한 전략)"""
    rng = np.random.RandomState(seed)
    
    # 인간 유형 (시뮬레이션)
    human_types = ['conditional_cooperator', 'free_rider', 'altruist', 'tit_for_tat', 'random']
    human_type = human_types[seed % len(human_types)]
    
    ai_agent = AIPartner(condition_ai)
    
    human_contribs = []
    ai_contribs = []
    payoffs = []
    prev_avg = 50
    
    for r in range(1, N_ROUNDS + 1):
        resource = 0.5 + 0.01 * (r - 10)
        
        # AI 결정
        ai_c = ai_agent.decide(r, prev_avg, resource, ENDOWMENT)
        
        # 인간 결정 (유형별)
        if human_type == 'conditional_cooperator':
            human_c = int(np.clip(prev_avg * 0.9 + rng.normal(0, 5), 0, ENDOWMENT))
        elif human_type == 'free_rider':
            human_c = int(np.clip(rng.normal(10, 5), 0, ENDOWMENT))
        elif human_type == 'altruist':
            human_c = int(np.clip(rng.normal(70, 10), 0, ENDOWMENT))
        elif human_type == 'tit_for_tat':
            human_c = int(np.clip(ai_c + rng.normal(0, 5), 0, ENDOWMENT))
        else:
            human_c = rng.randint(0, ENDOWMENT + 1)
        
        human_contribs.append(human_c)
        ai_contribs.append(ai_c)
        
        # 보수 계산 (3 AI + 1 Human)
        total = human_c + ai_c * 3
        share = total * MULTIPLIER / 4
        payoff = ENDOWMENT - human_c + share
        payoffs.append(payoff)
        
        prev_avg = (human_c + ai_c * 3) / 4
    
    return {
        "human_type": human_type,
        "ai_condition": condition_ai,
        "human_contribs": human_contribs,
        "ai_contribs": ai_contribs,
        "payoffs": payoffs,
        "mean_human_contrib": float(np.mean(human_contribs)),
        "mean_ai_contrib": float(np.mean(ai_contribs)),
        "total_payoff": float(np.sum(payoffs)),
    }


def run_pilot_simulation():
    """10명 × 3조건 시뮬레이션"""
    results = {}
    for condition in AI_CONDITIONS:
        runs = [simulate_human(condition, s) for s in range(N_SIMULATED_HUMANS)]
        results[condition] = {
            "runs": runs,
            "mean_human_contrib": float(np.mean([r["mean_human_contrib"] for r in runs])),
            "mean_ai_contrib": float(np.mean([r["mean_ai_contrib"] for r in runs])),
            "mean_payoff": float(np.mean([r["total_payoff"] for r in runs])),
        }
    return results


def plot_fig47(results):
    """Fig 47: 인간-AI 협력 패턴"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Fig 47: Human-AI Cooperation Patterns in PGG (Simulated Pilot, N=10)",
                 fontsize=14, fontweight='bold', y=0.98)
    
    colors_ai = {'selfish': '#e53935', 'prosocial': '#1e88e5', 'meta_ranking': '#43a047'}
    
    # 상단: 라운드별 기여율
    for j, condition in enumerate(AI_CONDITIONS):
        ax = axes[0, j]
        runs = results[condition]["runs"]
        
        human_avg = np.mean([r["human_contribs"] for r in runs], axis=0)
        human_std = np.std([r["human_contribs"] for r in runs], axis=0)
        ai_avg = np.mean([r["ai_contribs"] for r in runs], axis=0)
        
        rounds = range(1, N_ROUNDS + 1)
        ax.fill_between(rounds, human_avg - human_std, human_avg + human_std, alpha=0.2, color='#ff9800')
        ax.plot(rounds, human_avg, 'o-', color='#ff9800', markersize=3, linewidth=2, label='Human')
        ax.plot(rounds, ai_avg, 's-', color=colors_ai[condition], markersize=3, linewidth=2, label=f'AI ({condition})')
        
        ax.set_title(f'AI: {condition.replace("_", " ").title()}', fontweight='bold')
        ax.set_xlabel('Round'); ax.set_ylabel('Contribution')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)
    
    # 하단 좌: 조건별 평균 기여 비교
    ax = axes[1, 0]
    x = np.arange(len(AI_CONDITIONS))
    human_means = [results[c]["mean_human_contrib"] for c in AI_CONDITIONS]
    ai_means = [results[c]["mean_ai_contrib"] for c in AI_CONDITIONS]
    ax.bar(x - 0.15, human_means, 0.3, label='Human', color='#ff9800')
    ax.bar(x + 0.15, ai_means, 0.3, label='AI', color='#1e88e5')
    ax.set_title('Mean Contribution by Condition', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([c.replace('_', '\n') for c in AI_CONDITIONS], fontsize=8)
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    
    # 하단 중: 보수 분포
    ax = axes[1, 1]
    payoff_data = [[r["total_payoff"] for r in results[c]["runs"]] for c in AI_CONDITIONS]
    bp = ax.boxplot(payoff_data, labels=[c[:8] for c in AI_CONDITIONS], patch_artist=True)
    for patch, color in zip(bp['boxes'], [colors_ai[c] for c in AI_CONDITIONS]):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    ax.set_title('Total Payoff Distribution', fontweight='bold')
    ax.set_ylabel('Total Points'); ax.grid(True, axis='y', alpha=0.3)
    
    # 하단 우: 인간 유형별 기여
    ax = axes[1, 2]
    human_types = set()
    for c in AI_CONDITIONS:
        for r in results[c]["runs"]:
            human_types.add(r["human_type"])
    human_types = sorted(human_types)
    
    for c in AI_CONDITIONS:
        type_means = []
        for ht in human_types:
            hm = [r["mean_human_contrib"] for r in results[c]["runs"] if r["human_type"] == ht]
            type_means.append(np.mean(hm) if hm else 0)
        ax.plot(human_types, type_means, 'o-', color=colors_ai[c], linewidth=2, markersize=6, label=c)
    
    ax.set_title('Human Type × AI Condition', fontweight='bold')
    ax.set_xticklabels(human_types, rotation=30, fontsize=7)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, "fig47_human_ai_cooperation.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[O8] Fig 47 저장: {path}")


def plot_fig48(results):
    """Fig 48: 인간 적응 패턴"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Fig 48: Human Adaptation to AI Partners",
                 fontsize=14, fontweight='bold', y=1.02)
    
    colors_ai = {'selfish': '#e53935', 'prosocial': '#1e88e5', 'meta_ranking': '#43a047'}
    
    # 좌: 전반/후반 기여 변화
    ax = axes[0]
    x = np.arange(len(AI_CONDITIONS))
    early_means = []
    late_means = []
    for c in AI_CONDITIONS:
        early = [np.mean(r["human_contribs"][:5]) for r in results[c]["runs"]]
        late = [np.mean(r["human_contribs"][-5:]) for r in results[c]["runs"]]
        early_means.append(np.mean(early))
        late_means.append(np.mean(late))
    
    ax.bar(x - 0.15, early_means, 0.3, label='Early (R1-5)', color='#90caf9')
    ax.bar(x + 0.15, late_means, 0.3, label='Late (R16-20)', color='#1565c0')
    ax.set_title('Adaptation: Early vs Late Rounds', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([c.replace('_', '\n') for c in AI_CONDITIONS], fontsize=8)
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    
    # 중: 기여 추세선
    ax = axes[1]
    for c in AI_CONDITIONS:
        all_contribs = np.mean([r["human_contribs"] for r in results[c]["runs"]], axis=0)
        rounds = np.arange(1, N_ROUNDS + 1)
        z = np.polyfit(rounds, all_contribs, 1)
        trend = np.poly1d(z)
        ax.plot(rounds, all_contribs, 'o', color=colors_ai[c], markersize=3, alpha=0.6)
        ax.plot(rounds, trend(rounds), '-', color=colors_ai[c], linewidth=2, label=f'{c} (slope={z[0]:+.2f})')
    ax.set_title('Contribution Trend Lines', fontweight='bold')
    ax.set_xlabel('Round'); ax.set_ylabel('Human Contribution')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # 우: 인간-AI 상관관계
    ax = axes[2]
    for c in AI_CONDITIONS:
        h_all = []; a_all = []
        for r in results[c]["runs"]:
            h_all.extend(r["human_contribs"])
            a_all.extend(r["ai_contribs"])
        ax.scatter(a_all, h_all, alpha=0.15, s=10, color=colors_ai[c], label=c)
        z = np.polyfit(a_all, h_all, 1)
        ax.plot(sorted(set(a_all)), np.poly1d(z)(sorted(set(a_all))), color=colors_ai[c], linewidth=2)
    ax.set_title('Human vs AI Contribution Correlation', fontweight='bold')
    ax.set_xlabel('AI Contribution'); ax.set_ylabel('Human Contribution')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig48_human_adaptation.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[O8] Fig 48 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [O8] Human-AI 파일럿 실험 분석 (시뮬레이션)")
    print("=" * 60)
    
    results = run_pilot_simulation()
    
    print("\n--- Pilot Summary ---")
    for condition in AI_CONDITIONS:
        d = results[condition]
        print(f"  {condition:>14s} | Human={d['mean_human_contrib']:.1f} | "
              f"AI={d['mean_ai_contrib']:.1f} | Payoff={d['mean_payoff']:.0f}")
    
    plot_fig47(results)
    plot_fig48(results)
    
    json_data = {c: {k: v for k, v in d.items() if k != "runs"} for c, d in results.items()}
    json_path = os.path.join(OUTPUT_DIR, "pilot_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\n[O8] 결과 JSON: {json_path}")
