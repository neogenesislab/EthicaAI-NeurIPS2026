"""
EthicaAI L4: Constitutional AI Agent (고도화)
Phase L — LLM Mock 제거 → 규칙 기반 추론 엔진

Sen의 메타-랭킹을 5단계 의사결정 트리로 구현합니다.
API 키 없이도 작동하며, PGG/IPD/Cleanup 3개 환경에서 테스트합니다.
"""
import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class ConstitutionalRule:
    """헌법적 추론 규칙."""
    name: str
    priority: int  # 낮을수록 높은 우선순위
    condition: str  # 조건 설명 (디버그용)
    
    def evaluate(self, state: dict) -> bool:
        raise NotImplementedError


class SurvivalRule(ConstitutionalRule):
    """P1: 생존 규칙 — 자원 위기 시 자기보존 최우선."""
    def __init__(self):
        super().__init__("Survival Override", 1, "resource < survival_threshold")
    
    def evaluate(self, state):
        return state.get('resource_level', 5.0) < 2.0


class AbundanceCommitment(ConstitutionalRule):
    """P2: 풍요 헌신 — 여유 있을 때 사회적 기여."""
    def __init__(self):
        super().__init__("Abundance Commitment", 2, "resource > abundance_threshold")
    
    def evaluate(self, state):
        return state.get('resource_level', 5.0) > 8.0


class ReciprocityRule(ConstitutionalRule):
    """P3: 호혜성 — 상대방 협력 이력 기반."""
    def __init__(self):
        super().__init__("Reciprocity", 3, "partner_cooperation_rate > 0.5")
    
    def evaluate(self, state):
        return state.get('partner_coop_rate', 0.5) > 0.5


class FairnessRule(ConstitutionalRule):
    """P4: 공정성 — 불평등이 심할 때 재분배."""
    def __init__(self):
        super().__init__("Fairness Correction", 4, "gini > fairness_threshold")
    
    def evaluate(self, state):
        return state.get('gini', 0.3) > 0.4


class DefaultRule(ConstitutionalRule):
    """P5: 기본값 — SVO 기반 기본 행동."""
    def __init__(self):
        super().__init__("SVO Default", 5, "always true")
    
    def evaluate(self, state):
        return True


class ConstitutionalAgent:
    """헌법적 추론 에이전트.
    
    5단계 우선순위 기반 의사결정:
    P1(생존) > P2(풍요 헌신) > P3(호혜성) > P4(공정성) > P5(SVO 기본)
    """
    
    def __init__(self, agent_id, svo_theta=45, use_meta=True):
        self.agent_id = agent_id
        self.svo_theta = svo_theta
        self.use_meta = use_meta
        self.history = []
        
        self.constitution = [
            SurvivalRule(),
            AbundanceCommitment(),
            ReciprocityRule(),
            FairnessRule(),
            DefaultRule(),
        ]
    
    def decide(self, state: dict) -> dict:
        """헌법 기반 의사결정."""
        # SVO 기본 협력 확률
        base_coop = np.sin(np.radians(self.svo_theta))
        
        if not self.use_meta:
            # 메타-랭킹 없으면 SVO만 사용
            action = 'cooperate' if np.random.random() < base_coop else 'defect'
            return {'action': action, 'rule': 'SVO Only', 'lambda': base_coop, 'reasoning': []}
        
        # 헌법적 추론 체인
        reasoning = []
        coop_prob = base_coop
        triggered_rule = None
        
        for rule in self.constitution:
            triggered = rule.evaluate(state)
            reasoning.append({
                'rule': rule.name,
                'priority': rule.priority,
                'triggered': triggered,
                'condition': rule.condition,
            })
            
            if triggered and triggered_rule is None:
                triggered_rule = rule
                
                if isinstance(rule, SurvivalRule):
                    coop_prob = 0.05  # 거의 이기적
                elif isinstance(rule, AbundanceCommitment):
                    coop_prob = min(1.0, base_coop * 1.5)
                elif isinstance(rule, ReciprocityRule):
                    partner_rate = state.get('partner_coop_rate', 0.5)
                    coop_prob = base_coop * (0.5 + partner_rate)
                elif isinstance(rule, FairnessRule):
                    coop_prob = min(1.0, base_coop + 0.2)
        
        action = 'cooperate' if np.random.random() < coop_prob else 'defect'
        
        decision = {
            'action': action,
            'rule': triggered_rule.name if triggered_rule else 'None',
            'lambda': float(coop_prob),
            'reasoning': reasoning,
        }
        self.history.append(decision)
        return decision


def test_pgg(theta, use_meta, n_rounds=10, n_agents=4, seed=42):
    """PGG 환경 테스트."""
    rng = np.random.RandomState(seed)
    
    endowment = 20
    multiplier = 1.6
    agents = [ConstitutionalAgent(f"a{i}", theta, use_meta) for i in range(n_agents)]
    
    contributions = []
    for t in range(n_rounds):
        round_contribs = []
        for agent in agents:
            state = {
                'resource_level': 5.0 + rng.normal(0, 2),
                'partner_coop_rate': np.mean(contributions[-1]) / endowment if contributions else 0.5,
                'gini': 0.3,
                'round': t,
            }
            decision = agent.decide(state)
            contrib = endowment * decision['lambda'] if decision['action'] == 'cooperate' else endowment * 0.1
            round_contribs.append(contrib)
        contributions.append(round_contribs)
    
    rates = [np.mean(c) / endowment for c in contributions]
    return rates


def test_ipd(theta, use_meta, n_rounds=100, seed=42):
    """IPD 환경 테스트."""
    rng = np.random.RandomState(seed)
    a1 = ConstitutionalAgent("a1", theta, use_meta)
    a2 = ConstitutionalAgent("a2", theta, use_meta)
    
    coop_history = []
    a1_coop_count, a2_coop_count = 0, 0
    
    for t in range(n_rounds):
        s1 = {'resource_level': 5.0 + rng.normal(0, 1), 'partner_coop_rate': a2_coop_count / max(1, t), 'gini': 0.3}
        s2 = {'resource_level': 5.0 + rng.normal(0, 1), 'partner_coop_rate': a1_coop_count / max(1, t), 'gini': 0.3}
        
        d1, d2 = a1.decide(s1), a2.decide(s2)
        c1 = 1 if d1['action'] == 'cooperate' else 0
        c2 = 1 if d2['action'] == 'cooperate' else 0
        a1_coop_count += c1
        a2_coop_count += c2
        coop_history.append((c1 + c2) / 2)
    
    return coop_history


def test_cleanup(theta, use_meta, n_steps=200, n_agents=10, seed=42):
    """Cleanup 환경 간이 테스트."""
    rng = np.random.RandomState(seed)
    agents = [ConstitutionalAgent(f"a{i}", theta, use_meta) for i in range(n_agents)]
    
    resource = 50.0
    max_resource = 50.0
    clean_history = []
    
    for t in range(n_steps):
        clean_count = 0
        for agent in agents:
            state = {
                'resource_level': resource / max_resource * 10,
                'partner_coop_rate': np.mean(clean_history[-10:]) if clean_history else 0.5,
                'gini': 0.3,
            }
            decision = agent.decide(state)
            if decision['action'] == 'cooperate':
                clean_count += 1
                resource += 0.5  # 청소하면 자원 증가
            else:
                resource -= 0.3  # 수확하면 자원 감소
        
        resource = max(0, min(max_resource, resource))
        clean_history.append(clean_count / n_agents)
    
    return clean_history


def plot_constitutional(results, output_dir):
    """Figure 23: Constitutional AI Agent 결과."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Fig 23. Constitutional AI Agent: Rule-Based Meta-Ranking", fontsize=14, fontweight='bold')
    
    # 1. PGG 기여율
    ax = axes[0]
    for label, data, color, dash in [
        ('Meta (45°)', results['pgg_meta_45'], '#4fc3f7', '-'),
        ('Base (45°)', results['pgg_base_45'], '#888', '--'),
        ('Meta (15°)', results['pgg_meta_15'], '#ce93d8', '-'),
        ('Base (15°)', results['pgg_base_15'], '#aaa', '--'),
    ]:
        ax.plot(data, linestyle=dash, color=color, label=label, linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Contribution Rate')
    ax.legend(fontsize=8)
    ax.set_title('(A) PGG Contribution')
    ax.set_ylim(0, 1.1)
    
    # 2. IPD 협력률
    ax = axes[1]
    window = 10
    for label, data, color, dash in [
        ('Meta', results['ipd_meta'], '#4fc3f7', '-'),
        ('Baseline', results['ipd_base'], '#888', '--'),
    ]:
        smooth = np.convolve(data, np.ones(window)/window, mode='valid')
        ax.plot(smooth, linestyle=dash, color=color, label=label, linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Cooperation Rate (MA-10)')
    ax.legend()
    ax.set_title('(B) IPD Cooperation')
    ax.set_ylim(0, 1.1)
    
    # 3. Cleanup 보전율
    ax = axes[2]
    window = 20
    for label, data, color, dash in [
        ('Meta', results['cleanup_meta'], '#66bb6a', '-'),
        ('Baseline', results['cleanup_base'], '#888', '--'),
    ]:
        smooth = np.convolve(data, np.ones(window)/window, mode='valid')
        ax.plot(smooth, linestyle=dash, color=color, label=label, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Clean Action Rate')
    ax.legend()
    ax.set_title('(C) Cleanup Conservation')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fig23_constitutional.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[L4] Figure 저장: {out_path}")
    return out_path


def print_reasoning_example(agent):
    """추론 체인 예시 출력."""
    if agent.history:
        last = agent.history[-1]
        print(f"\n--- Constitutional Reasoning Chain ---")
        print(f"  Action: {last['action']} (λ={last['lambda']:.3f})")
        print(f"  Triggered Rule: {last['rule']}")
        for step in last['reasoning']:
            marker = "✓" if step['triggered'] else "✗"
            print(f"    [{marker}] P{step['priority']} {step['rule']}: {step['condition']}")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "simulation/outputs/reproduce"
    os.makedirs(output_dir, exist_ok=True)
    
    print("[L4] Constitutional AI Agent 고도화 실험 시작...")
    
    results = {
        'pgg_meta_45': test_pgg(45, True),
        'pgg_base_45': test_pgg(45, False),
        'pgg_meta_15': test_pgg(15, True),
        'pgg_base_15': test_pgg(15, False),
        'ipd_meta': test_ipd(45, True),
        'ipd_base': test_ipd(45, False),
        'cleanup_meta': test_cleanup(45, True),
        'cleanup_base': test_cleanup(45, False),
    }
    
    plot_constitutional(results, output_dir)
    
    # 추론 체인 예시
    agent = ConstitutionalAgent("demo", 45, True)
    for scenario, state in [
        ("Starvation", {'resource_level': 1.0, 'partner_coop_rate': 0.8, 'gini': 0.2}),
        ("Normal", {'resource_level': 5.0, 'partner_coop_rate': 0.6, 'gini': 0.3}),
        ("Abundance", {'resource_level': 9.0, 'partner_coop_rate': 0.4, 'gini': 0.5}),
    ]:
        print(f"\n=== Scenario: {scenario} ===")
        decision = agent.decide(state)
        print_reasoning_example(agent)
    
    # JSON 저장
    json_path = os.path.join(output_dir, 'constitutional_results.json')
    save_data = {
        'pgg_meta_45_avg': float(np.mean(results['pgg_meta_45'])),
        'pgg_base_45_avg': float(np.mean(results['pgg_base_45'])),
        'ipd_meta_avg': float(np.mean(results['ipd_meta'][-20:])),
        'ipd_base_avg': float(np.mean(results['ipd_base'][-20:])),
        'cleanup_meta_avg': float(np.mean(results['cleanup_meta'][-50:])),
        'cleanup_base_avg': float(np.mean(results['cleanup_base'][-50:])),
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[L4] 결과 JSON: {json_path}")
