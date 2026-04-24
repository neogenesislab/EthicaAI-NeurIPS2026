"""
OptimalAgent — 최선합리성 에이전트 (Optimal Rationality)

센의 메타 랭킹(Meta-Ranking) 이론을 구현.
단일 보상(U_self)이 아닌, 상황에 따라 이기적 모드와 헌신 모드 사이를 전환하거나 가중치를 조절.

보고서 수식 §2.2:
    R_total = (1 - λ)·U_self + λ·[U_meta - ψ(s)]
    
    여기서:
    - λ (lambda): 헌신 수준 (Commitment Level)
    - U_meta: 사회적/윤리적 효용
    - ψ(s): 자제 비용 (Self-Control Cost)

이 에이전트는 내부적으로 두 가지 가치 평가 기준을 가지고 행동을 선택함.
"""
import numpy as np
from typing import Dict, Any

from simulation.agents.base_agent import BaseAgent
from simulation.config import MetaRankingParams, RewardWeights


class OptimalAgent(BaseAgent):
    """최선합리성 에이전트."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        
        # 메타 랭킹 파라미터 로드
        self.meta_params = MetaRankingParams()
        # 보상 가중치 (U* 계산용)
        self.weights = RewardWeights()
        
        # 동적 헌신 수준 (초기값)
        self.commitment_lambda = self.meta_params.commitment_lambda
        self.base_lambda = self.commitment_lambda # 기준 헌신 수준
        
        # 상태
        self.current_mode = "neutral" 
        self.wealth = 100.0 # 초기 자원 (생존 임계값 테스트용)
        self.survival_threshold = 50.0 # 생존 임계값

    def update_commitment(self):
        """자원 상태에 따른 헌신 수준 동적 조절.
        
        가설: 곳간에서 인심 난다 (Aristotle/Maslow).
        - 자원이 풍족하면(>150%) 헌신 수준 증가.
        - 생존 위협 시(<50%) 헌신 수준 급격히 감소.
        """
        if self.wealth < self.survival_threshold:
            # 생존 모드: 이기적 본능 우선
            target_lambda = 0.0
        elif self.wealth > self.survival_threshold * 2:
            # 풍요 모드: 헌신 강화
            target_lambda = min(1.0, self.base_lambda * 1.5)
        else:
            # 일반 모드
            target_lambda = self.base_lambda
            
        # 부드러운 변화 (EMA)
        self.commitment_lambda = 0.9 * self.commitment_lambda + 0.1 * target_lambda

    def update(self, reward: float, info: Dict[str, Any] = None):
        """상태 업데이트 (자원 누적 및 헌신 조절)."""
        super().update(reward, info)
        self.wealth += reward
        self.update_commitment()

    def act(self, observation: np.ndarray) -> int:
        """행동 선택 (메타 랭킹 + 자제 비용 + 처벌)."""
        # 관측: [상대_C, 상대_D, 상대_P, 초기/노이즈]
        
        # 1. 상황 인식
        opponent_defected = (observation[1] == 1.0)
        
        # 2. 효용 추정 (기본 행동: C=0, D=1)
        u_self = [0.0, 1.0] # [C, D]
        u_meta = [1.0, 0.0] # [C, D]
        psi = [0.3, 0.0]    # [C, D] - 자제 비용
        
        r_total = []
        for a in [0, 1]:
            val = (1 - self.commitment_lambda) * u_self[a] + \
                  self.commitment_lambda * (u_meta[a] - psi[a])
            r_total.append(val)
        
        base_action = int(np.argmax(r_total)) # 0 or 1
        
        # 3. 처벌 결정 (Sanction)
        # 처벌은 배신자에게 정의를 구현하는 것 (Strong Reciprocity)
        # 조건: 상대가 배신했고(opponent_defected), 내가 헌신적일 때(commitment_lambda > 0.3)
        # 처벌 행동으로 변환: 0->2, 1->3
        
        do_punish = False
        if opponent_defected and self.commitment_lambda > 0.3:
            # 처벌 효용 > 처벌 비용?
            # 메타 랭킹에서는 "정의 구현"에 높은 가치 부여
            # u_punish = lambda * 2.0 (정의감) - (1-lambda) * 1.0 (비용)
            # config 파일을 직접 참조하기 어려우므로 하드코딩된 가치 사용 (추후 개선)
            punish_utility = self.commitment_lambda * 2.0 - (1 - self.commitment_lambda) * 1.0
            if punish_utility > 0:
                do_punish = True
        
        final_action = base_action
        if do_punish:
            final_action = base_action + 2 # 0->2, 1->3
            
        # 모드 기록
        mode_str = "commitment" if base_action == 0 else "selfish"
        if do_punish:
            mode_str += "+punish"
        self.current_mode = f"{mode_str} (λ={self.commitment_lambda:.2f})"
            
        return final_action

    def set_commitment(self, lambda_val: float):
        """헌신 수준 조절 (학습 또는 실험용)."""
        self.commitment_lambda = np.clip(lambda_val, 0.0, 1.0)
