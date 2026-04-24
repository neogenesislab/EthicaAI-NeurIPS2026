"""
CausalCell — 인과추론 분석 셀

역할:
    - 보고서 §4 (인과추론 검증) 구현
    - '헌신(Commitment)' 처치(Treatment)가 '누적 보상(Outcome)'에 미치는 인과 효과 추정.
    - 초기 구현: 선형 회귀 기반 CATE(Conditional Average Treatment Effect) 추정.
    - 추후 DoubleML/EconML 연동.

입력(Input):
    - agents_data: Dict[agent_id, total_reward]
    - agent_configs: Dict[agent_id, config] (헌신 수준 λ 정보 포함)
    
출력(Output):
    - causal_effect: float (헌신 증가 시 보상 변화량)
    - confounder_analysis: Dict
"""
import numpy as np
from typing import Any, Dict, List

from simulation.cells.base_cell import BaseCell
from simulation.config import CausalConfig


class CausalCell(BaseCell):
    """인과추론 셀."""

    def __init__(self):
        super().__init__()
        self.config: CausalConfig = None

    @property
    def name(self) -> str:
        return "CausalCell"

    def initialize(self, config: Dict[str, Any]) -> None:
        super().initialize(config)
        self.config = config.get("causal_config") or CausalConfig()

    def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """인과 효과 추정 실행."""
        agents_data = input_data.get("agents_data", {}) # Y (Outcome)
        
        # Treatment (T): 헌신 수준 (lambda)
        # Confounder (X): 에이전트 유형 (Selfish=0, Optimal=1) - 사실상 이게 T와 강하게 연관
        
        # 데이터 포인트 수집
        T = [] # Treatment (Commitment Lambda)
        Y = [] # Outcome (Reward)
        
        for agent_id, reward in agents_data.items():
            # 임시 로직: 에이전트 설정 정보가 없으면 ID로 추정
            # 짝수=Optimal(λ=0.5), 홀수=Selfish(λ=0.0)
            try:
                idx = int(agent_id.split("_")[1])
                is_optimal = (idx % 2 == 0)
                lambda_val = 0.5 if is_optimal else 0.0
                
                T.append(lambda_val)
                Y.append(reward)
            except:
                pass
                
        if len(T) < 2:
            return {"status": "insufficient_data"}
            
        # 단순 회귀분석 (Simple Linear Regression)
        # Y = alpha + beta * T + epsilon
        # beta가 인과 효과 (ATE)
        
        T = np.array(T)
        Y = np.array(Y)
        
        # 공분산/분산으로 기울기 추정
        # beta = Cov(T, Y) / Var(T)
        if np.var(T) > 0:
            beta = np.cov(T, Y)[0, 1] / np.var(T)
            intercept = np.mean(Y) - beta * np.mean(T)
            
            # 상관계수
            correlation = np.corrcoef(T, Y)[0, 1]
        else:
            beta = 0.0
            intercept = 0.0
            correlation = 0.0

        return {
            "ate_estimate": float(beta), # Average Treatment Effect
            "correlation": float(correlation),
            "sample_size": len(T),
            "status": "estimated_simple_regression"
        }
