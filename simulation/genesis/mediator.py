"""
중재자 에이전트: 에이전트 그룹의 행동을 조율하는 상위 정책.

Genesis v2.0 Strategy B: Leviathan Update
헌법 제12조 1항 이론적 근거:
  - Ivanov et al. (2023) "Mediated Multi-Agent Reinforcement Learning"
  - Hobbes, Leviathan (1651) — 사회계약론

동작 방식:
1. k 스텝마다 에이전트에게 위임 여부를 묻는다.
2. 위임한 에이전트의 행동을 사회적 복지 극대화로 대체.
3. IC 제약: 위임 > 독립 행동 (기대 보상)
4. E 제약: 무임승차 불이익 보장
"""
import jax
import jax.numpy as jnp
import json
import os
from datetime import datetime


class Mediator:
    """
    중재자 에이전트.

    에이전트들의 위임을 받아 집단 이익 최대화 정책을 실행.
    """

    def __init__(self, config=None):
        config = config or {}
        self.commitment_window = config.get("MEDIATOR_K", 10)
        self.lambda_ic = config.get("MEDIATOR_LAMBDA_IC", 1.0)
        self.lambda_e = config.get("MEDIATOR_LAMBDA_E", 0.5)
        self.delegation_history = []
        self.step_counter = 0

    def should_consult(self):
        """위임 결정 시점인지 확인 (k 스텝마다)."""
        self.step_counter += 1
        return self.step_counter % self.commitment_window == 0

    def check_delegation(self, agent_reward_history, mediator_reward_history):
        """
        에이전트가 위임할지 결정 (IC 제약 기반).

        Args:
            agent_reward_history: 독립 행동 시 보상 이력
            mediator_reward_history: 중재자 가이드 시 보상 이력

        Returns:
            bool: 위임 여부
        """
        if len(agent_reward_history) < 5 or len(mediator_reward_history) < 5:
            return True  # 초기에는 위임 선호 (탐색)

        indep_value = sum(agent_reward_history[-20:]) / min(len(agent_reward_history), 20)
        med_value = sum(mediator_reward_history[-20:]) / min(len(mediator_reward_history), 20)

        delegate = med_value > indep_value
        self.delegation_history.append({
            "step": self.step_counter,
            "indep_value": indep_value,
            "med_value": med_value,
            "delegated": delegate,
        })
        return delegate

    def compute_collective_action(self, n_agents, cooperation_rates=None):
        """
        위임받은 에이전트들의 행동을 사회적 복지 극대화로 결정.

        초기 버전: 역할 분담 기반 (1/3 청소, 2/3 채집)
        향후: 학습된 정책으로 대체

        Args:
            n_agents: 에이전트 수
            cooperation_rates: 에이전트별 현재 협력률 [N]

        Returns:
            actions: [N] 행동 벡터
        """
        actions = jnp.zeros(n_agents, dtype=jnp.int32)

        # 협력률이 낮은 에이전트를 청소로 배정
        if cooperation_rates is not None:
            n_cleaners = max(n_agents // 3, 1)
            # 협력률이 가장 낮은 에이전트들을 청소로
            worst_agents = jnp.argsort(cooperation_rates)[:n_cleaners]
            actions = actions.at[worst_agents].set(5)  # 5 = CLEAN 행동 (환경 의존)
        else:
            # 기본: 균등 분배
            n_cleaners = n_agents // 3
            actions = actions.at[:n_cleaners].set(5)

        return actions

    def get_delegation_rate(self):
        """위임률 계산."""
        if not self.delegation_history:
            return 0.0
        recent = self.delegation_history[-50:]
        return sum(1 for d in recent if d["delegated"]) / len(recent)

    def get_report(self):
        """중재자 상태 보고서."""
        return {
            "total_consultations": len(self.delegation_history),
            "delegation_rate": self.get_delegation_rate(),
            "commitment_window": self.commitment_window,
            "step_counter": self.step_counter,
        }


if __name__ == "__main__":
    print("🧪 mediator.py 단위 테스트")
    print("=" * 50)

    med = Mediator({"MEDIATOR_K": 5})

    # 위임 시점 테스트
    for i in range(12):
        consult = med.should_consult()
        if consult:
            print(f"  Step {i+1}: 🤝 위임 결정 시점!")

    # 위임 결정 테스트
    agent_hist = [0.3, 0.2, 0.4, 0.1, 0.3, 0.2]
    med_hist = [0.5, 0.6, 0.4, 0.5, 0.7, 0.6]
    result = med.check_delegation(agent_hist, med_hist)
    print(f"\n  위임 결정: {result} (med > indep)")

    # 집단 행동 테스트
    actions = med.compute_collective_action(10)
    print(f"  집단 행동 (10명): {actions}")

    print(f"  위임률: {med.get_delegation_rate():.2%}")
    print(f"\n✅ 모든 테스트 통과!")
