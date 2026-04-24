"""
Q3: 다중 도덕 이론 비교 실험
EthicaAI Phase Q — 신규 기여

4가지 도덕 이론을 λ 메커니즘으로 형식화하고 비교:
1. 공리주의 (Utilitarian): λ = argmax(group_welfare)
2. 의무론 (Deontological): λ = rule_based (min 기여 의무)
3. 덕 윤리 (Virtue Ethics): λ_t = slow habitual adaptation
4. 상황 윤리 (Situational, ours): λ_t = f(resource, SVO, history)

핵심 가설: "상황 윤리만이 진화적으로 안정하다"

출력: Fig 53 (이론별 성능 비교), Fig 54 (진화적 안정성 토너먼트)
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
N_EVOLUTION_GENS = 100
ENDOWMENT = 100


class MoralTheory:
    """도덕 이론 기반 클래스"""
    
    def __init__(self, name, svo_deg=45.0):
        self.name = name
        self.svo_theta = np.radians(svo_deg)
        self.lambda_t = np.sin(self.svo_theta)
        self.history = []
    
    def decide(self, resource, group_avg, t, rng):
        raise NotImplementedError


class Utilitarian(MoralTheory):
    """공리주의: 전체 복지 최대화 (탐욕적 최적화)"""
    
    def decide(self, resource, group_avg, t, rng):
        # 전체 복지 = 공공재 × 배율 → 기여 max
        # 단, 자기보존을 위한 최소 보유 (20%)
        optimal = ENDOWMENT * 0.8
        self.lambda_t = optimal / ENDOWMENT
        noise = rng.normal(0, 2)
        return int(np.clip(optimal + noise, 0, ENDOWMENT))


class Deontological(MoralTheory):
    """의무론: 규칙 기반 의무 기여 (Categorical Imperative)"""
    
    DUTY_RATE = 0.5  # 50% 의무 기여 (설정 분리)
    
    def decide(self, resource, group_avg, t, rng):
        # 규칙: 항상 50% 기여 (상황 무관)
        duty = ENDOWMENT * self.DUTY_RATE
        self.lambda_t = self.DUTY_RATE
        noise = rng.normal(0, 2)
        return int(np.clip(duty + noise, 0, ENDOWMENT))


class VirtueEthics(MoralTheory):
    """덕 윤리: 습관적 적응 (느린 학습)"""
    
    LEARNING_RATE = 0.01  # 매우 느린 적응 (설정 분리)
    
    def decide(self, resource, group_avg, t, rng):
        # 이웃 평균을 향해 천천히 수렴 (아리스토텔레스적 '습관')
        avg_norm = group_avg / ENDOWMENT
        self.lambda_t = self.lambda_t * (1 - self.LEARNING_RATE) + avg_norm * self.LEARNING_RATE
        contrib = self.lambda_t * ENDOWMENT * 0.8
        noise = rng.normal(0, 2)
        return int(np.clip(contrib + noise, 0, ENDOWMENT))


class SituationalEthics(MoralTheory):
    """상황 윤리 (EthicaAI): 자원-SVO 기반 동적 λ"""
    
    ALPHA = 0.1  # 조절 속도 (설정 분리)
    
    def decide(self, resource, group_avg, t, rng):
        base = np.sin(self.svo_theta)
        if resource < 0.2:
            target = max(0, base * 0.3)
        elif resource > 0.7:
            target = min(1, base * 1.5)
        else:
            reciprocity = group_avg / (ENDOWMENT + 1e-8)
            target = base * (0.7 + 0.6 * reciprocity)
        
        self.lambda_t = self.lambda_t * (1 - self.ALPHA) + target * self.ALPHA
        contrib = self.lambda_t * ENDOWMENT * 0.8
        noise = rng.normal(0, 2)
        return int(np.clip(contrib + noise, 0, ENDOWMENT))


class Selfish(MoralTheory):
    """이기주의 (Control): 무기여"""
    
    def decide(self, resource, group_avg, t, rng):
        self.lambda_t = 0.0
        return int(np.clip(rng.normal(5, 3), 0, ENDOWMENT))


THEORY_CLASSES = {
    "utilitarian": Utilitarian,
    "deontological": Deontological,
    "virtue": VirtueEthics,
    "situational": SituationalEthics,
    "selfish": Selfish,
}


def simulate_population(theory_name, seed):
    """단일 이론 인구 시뮬레이션"""
    rng = np.random.RandomState(seed)
    agents = [THEORY_CLASSES[theory_name](theory_name) for _ in range(N_AGENTS)]
    
    resource = 0.5
    coop_history = []
    resource_history = []
    welfare_history = []
    
    for t in range(N_STEPS):
        group_avg = np.mean([a.lambda_t for a in agents]) * ENDOWMENT * 0.8
        
        contributions = [a.decide(resource, group_avg, t, rng) for a in agents]
        total = sum(contributions)
        public_good = total * 1.6 / N_AGENTS
        
        payoffs = [(ENDOWMENT - c) + public_good for c in contributions]
        welfare = sum(payoffs) / N_AGENTS
        
        resource = np.clip(resource + 0.02 * (np.mean(contributions) / ENDOWMENT - 0.3), 0, 1)
        
        coop_history.append(np.mean([c > ENDOWMENT * 0.3 for c in contributions]))
        resource_history.append(resource)
        welfare_history.append(welfare)
    
    return {
        "coop_history": coop_history,
        "resource_history": resource_history,
        "welfare_history": welfare_history,
        "mean_coop": float(np.mean(coop_history[-30:])),
        "mean_welfare": float(np.mean(welfare_history[-30:])),
        "mean_resource": float(np.mean(resource_history[-30:])),
        "sustainability": float(np.mean([r > 0.1 for r in resource_history])),
    }


def evolutionary_tournament():
    """진화적 토너먼트: 어떤 도덕 이론이 살아남는가?"""
    theories = list(THEORY_CLASSES.keys())
    n_theories = len(theories)
    
    # 초기 비율: 균등
    proportions = np.ones(n_theories) / n_theories
    proportion_history = [proportions.copy()]
    
    for gen in range(N_EVOLUTION_GENS):
        # 각 이론의 적합도 계산 (N_AGENTS × proportion 혼합)
        fitnesses = np.zeros(n_theories)
        
        for i, theory in enumerate(theories):
            if proportions[i] < 0.01:
                fitnesses[i] = 0
                continue
            
            rng = np.random.RandomState(gen)
            n_this = max(1, int(N_AGENTS * proportions[i]))
            agents = [THEORY_CLASSES[theory](theory) for _ in range(n_this)]
            
            resource = 0.5
            total_payoff = 0
            
            for t in range(50):
                group_avg = np.mean([a.lambda_t for a in agents]) * ENDOWMENT * 0.8
                contributions = [a.decide(resource, group_avg, t, rng) for a in agents]
                public_good = sum(contributions) * 1.6 / max(1, len(agents))
                payoffs = [(ENDOWMENT - c) + public_good for c in contributions]
                total_payoff += np.mean(payoffs)
                resource = np.clip(resource + 0.02 * (np.mean(contributions) / ENDOWMENT - 0.3), 0, 1)
            
            fitnesses[i] = total_payoff
        
        # Replicator dynamics
        avg_fitness = np.sum(proportions * fitnesses)
        if avg_fitness > 0:
            proportions = proportions * fitnesses / avg_fitness
            proportions = np.clip(proportions, 0.001, None)
            proportions /= proportions.sum()
        
        proportion_history.append(proportions.copy())
    
    return theories, np.array(proportion_history)


def plot_fig53(results):
    """Fig 53: 도덕 이론별 성능 비교"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Fig 53: Comparative Analysis of Moral Theories in Social Dilemmas",
                 fontsize=14, fontweight='bold', y=0.98)
    
    theories = list(results.keys())
    theory_colors = {
        'utilitarian': '#ff9800', 'deontological': '#9c27b0',
        'virtue': '#00897b', 'situational': '#1e88e5', 'selfish': '#e53935'
    }
    
    # 상단 좌: 협력률 추이
    ax = axes[0, 0]
    for theory in theories:
        runs = results[theory]
        avg_coop = np.mean([r["coop_history"] for r in runs], axis=0)
        ax.plot(avg_coop, color=theory_colors[theory], linewidth=2, label=theory.capitalize())
    ax.set_title('Cooperation Rate Over Time', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Cooperation Rate')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # 상단 중: 자원 추이
    ax = axes[0, 1]
    for theory in theories:
        runs = results[theory]
        avg_res = np.mean([r["resource_history"] for r in runs], axis=0)
        ax.plot(avg_res, color=theory_colors[theory], linewidth=2, label=theory.capitalize())
    ax.set_title('Resource Level Over Time', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Resource')
    ax.axhline(0.1, color='red', linestyle=':', alpha=0.5, label='Crisis Threshold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # 상단 우: 복지 추이
    ax = axes[0, 2]
    for theory in theories:
        runs = results[theory]
        avg_w = np.mean([r["welfare_history"] for r in runs], axis=0)
        ax.plot(avg_w, color=theory_colors[theory], linewidth=2, label=theory.capitalize())
    ax.set_title('Social Welfare Over Time', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Welfare')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # 하단 좌: 최종 성능 바차트
    ax = axes[1, 0]
    x = np.arange(len(theories))
    coops = [np.mean([r["mean_coop"] for r in results[t]]) for t in theories]
    ax.bar(x, coops, color=[theory_colors[t] for t in theories], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_title('Final Cooperation Rate', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([t[:6] for t in theories], rotation=15)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 하단 중: 복지 바차트
    ax = axes[1, 1]
    welfares = [np.mean([r["mean_welfare"] for r in results[t]]) for t in theories]
    ax.bar(x, welfares, color=[theory_colors[t] for t in theories], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_title('Final Social Welfare', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([t[:6] for t in theories], rotation=15)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 하단 우: 지속가능성
    ax = axes[1, 2]
    sustain = [np.mean([r["sustainability"] for r in results[t]]) * 100 for t in theories]
    bars = ax.bar(x, sustain, color=[theory_colors[t] for t in theories], alpha=0.8, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, sustain):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=9)
    ax.set_title('Sustainability (% Steps Above Crisis)', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([t[:6] for t in theories], rotation=15)
    ax.set_ylim(0, 105); ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, "fig53_moral_theories.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[Q3] Fig 53 저장: {path}")


def plot_fig54(theories, proportion_history):
    """Fig 54: 진화적 토너먼트 — 어떤 도덕이 살아남는가?"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Fig 54: Evolutionary Tournament — Which Moral Theory Survives?",
                 fontsize=14, fontweight='bold', y=1.02)
    
    theory_colors = {
        'utilitarian': '#ff9800', 'deontological': '#9c27b0',
        'virtue': '#00897b', 'situational': '#1e88e5', 'selfish': '#e53935'
    }
    
    # 좌: 비율 추이
    ax = axes[0]
    for i, theory in enumerate(theories):
        ax.plot(proportion_history[:, i], color=theory_colors[theory], linewidth=2.5, label=theory.capitalize())
        ax.fill_between(range(len(proportion_history)), 0, proportion_history[:, i],
                        color=theory_colors[theory], alpha=0.05)
    ax.set_title('Population Proportion Over Generations', fontweight='bold')
    ax.set_xlabel('Generation'); ax.set_ylabel('Proportion')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 우: 최종 비율 파이차트
    ax = axes[1]
    final_props = proportion_history[-1]
    labels = [f'{t.capitalize()}\n({p:.1%})' for t, p in zip(theories, final_props)]
    colors_list = [theory_colors[t] for t in theories]
    
    # 1% 미만은 합치기
    threshold = 0.01
    significant = final_props >= threshold
    if not all(significant):
        other_prop = sum(final_props[~significant])
        labels_filtered = [l for l, s in zip(labels, significant) if s] + [f'Others\n({other_prop:.1%})']
        props_filtered = list(final_props[significant]) + [other_prop]
        colors_filtered = [c for c, s in zip(colors_list, significant) if s] + ['#bdbdbd']
    else:
        labels_filtered = labels
        props_filtered = list(final_props)
        colors_filtered = colors_list
    
    ax.pie(props_filtered, labels=labels_filtered, colors=colors_filtered,
           autopct='', startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax.set_title('Final Evolutionary Equilibrium', fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig54_evolutionary_tournament.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[Q3] Fig 54 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [Q3] 다중 도덕 이론 비교 실험")
    print("=" * 60)
    
    # 1. 성능 비교
    print("\n[1/2] 도덕 이론별 시뮬레이션...")
    results = {}
    for theory_name in THEORY_CLASSES:
        runs = [simulate_population(theory_name, s) for s in range(N_SEEDS)]
        results[theory_name] = runs
        m = np.mean([r["mean_coop"] for r in runs])
        w = np.mean([r["mean_welfare"] for r in runs])
        s = np.mean([r["sustainability"] for r in runs])
        print(f"  {theory_name:>14s} | Coop={m:.3f} | Welfare={w:.1f} | Sustain={s*100:.1f}%")
    
    plot_fig53(results)
    
    # 2. 진화적 토너먼트
    print("\n[2/2] 진화적 토너먼트...")
    theories, proportion_history = evolutionary_tournament()
    
    print(f"\n--- Final Evolutionary Equilibrium ---")
    for i, theory in enumerate(theories):
        print(f"  {theory:>14s}: {proportion_history[-1, i]:.1%}")
    
    plot_fig54(theories, proportion_history)
    
    json_data = {
        "performance": {t: {"mean_coop": float(np.mean([r["mean_coop"] for r in runs])),
                           "mean_welfare": float(np.mean([r["mean_welfare"] for r in runs])),
                           "sustainability": float(np.mean([r["sustainability"] for r in runs]))}
                       for t, runs in results.items()},
        "evolution_final": {t: float(proportion_history[-1, i]) for i, t in enumerate(theories)},
    }
    json_path = os.path.join(OUTPUT_DIR, "moral_theories_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\n[Q3] 결과 JSON: {json_path}")
