"""
BoundedAgent — 제한적 합리성 에이전트 (Bounded Rationality)

Herbert Simon의 이론을 반영.
인지적 한계, 정보 처리 비용, 불완전한 정보를 노이즈(Noise)로 모델링.
최적해를 찾기보다 '만족하기(Satisficing)' 전략을 사용.
"""
import numpy as np
from typing import Dict, Any

from simulation.agents.base_agent import BaseAgent


class BoundedAgent(BaseAgent):
    """제한적 합리성 에이전트."""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        # 노이즈 수준 (표준편차)
        agent_conf = config.get("agent_config")
        self.noise_std = agent_conf.bounded_noise_std if agent_conf else 0.5
        
    def act(self, observation: np.ndarray) -> int:
        """행동 선택.
        
        합리적 판단(U_self 극대화)에 노이즈 추가.
        처벌 가능 환경이면 처벌도 무작위로 선택될 수 있음 (실수/오판).
        """
        obs_dim = len(observation)
        # 관측 차원이 4이면 처벌 가능 환경 [Opp_C, Opp_D, Opp_P, Init]
        can_punish = (obs_dim >= 4) 
        
        if can_punish:
            # [C, D, C+P, D+P]
            # 기본 효용: D(1) > C(0)
            # 처벌 행동(2,3)은 비용이 들므로 기본 효용 낮음 (-1.0)
            base_utility = np.array([0.0, 1.0, -1.0, 0.0]) 
            noise = np.random.normal(0, self.noise_std, size=4)
        else:
            base_utility = np.array([0.0, 1.0]) # [C, D]
            noise = np.random.normal(0, self.noise_std, size=2)
            
        perceived_utility = base_utility + noise
        
        return int(np.argmax(perceived_utility))
