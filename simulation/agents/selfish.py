"""
SelfishAgent — 이기적 에이전트 (Rational Fool)

RCT(합리적 선택 이론)에 따라 자신의 물질적 보상(U_self)만을 극대화하려 함.
사회적 규범이나 타인의 이익은 고려하지 않음.
초기 구현: 항상 배신(Defect)을 하는 전략 (Tit-for-Tat 등 전략 학습 전 단계)
"""
import numpy as np
from typing import Dict, Any

from simulation.agents.base_agent import BaseAgent


class SelfishAgent(BaseAgent):
    """이기적 에이전트."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.strategy = "always_defect"  # 기본 전략

    def act(self, observation: np.ndarray) -> int:
        """행동 선택.
        
        죄수의 딜레마에서:
        0 = 협력(Cooperate)
        1 = 배신(Defect)
        
        * 처벌(Sanction)이 활성화된 환경(Action Space 4)이라도
          이기적 에이전트는 비용이 드는 처벌(2, 3)을 하지 않음.
          따라서 항상 0 또는 1만 반환.
        """
        # 관측: [상대_C, 상대_D, 상대_P, 초기]
        
        # 1. 항상 배신 전략 (Nash Equilibrium for One-Shot PD)
        if self.strategy == "always_defect":
            return 1
        
        # 2. 랜덤 전략 (탐색용)
        elif self.strategy == "random":
            return np.random.randint(0, 2)
            
        # 3. Tit-for-Tat (눈에는 눈, 이에는 이) - 이기적이지만 호혜적
        elif self.strategy == "tit_for_tat":
            # 상대가 협력했으면 협력, 아니면 배신
            # 관측 구조가 [Opp_C, Opp_D, ...] 이므로
            # 초기 상태(obs[-1]==1)면 협력
            if observation[-1] == 1.0:
                return 0
            if observation[0] == 1.0: # 상대 협력
                return 0
            return 1 # 상대 배신
            
        return 1

    def set_strategy(self, strategy: str):
        self.strategy = strategy
