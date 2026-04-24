"""
EvalCell — 평가 및 통계 분석 셀

역할:
    - 실험 결과 데이터(trajectories)를 기반으로 성과 지표 계산
    - 보고서 §6.2 가설 검증 (Optimal vs Selfish)
    - 통계적 유의성 검정 (t-test 등)

입력(Input):
    - result_data: Dict[agent_id, total_reward]
    - trajectories: List[Dict]

출력(Output):
    - metrics: Dict (협력률, 평균 보상, 지니 계수)
    - hypothesis_test: Dict (검정 통계량, p-value)
"""
import numpy as np
from scipy import stats
from typing import Any, Dict, List

from simulation.cells.base_cell import BaseCell
from simulation.config import StatisticsConfig


class EvalCell(BaseCell):
    """실험 결과 평가 셀."""
    
    def __init__(self):
        super().__init__()
        self.config: StatisticsConfig = None

    @property
    def name(self) -> str:
        return "EvalCell"

    def initialize(self, config: Dict[str, Any]) -> None:
        super().initialize(config)
        self.config = config.get("stats_config") or StatisticsConfig()

    def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """평가 실행."""
        trajectories = input_data.get("trajectories", [])
        agents_data = input_data.get("agents_data", {}) # agent_id -> total_reward
        
        if not trajectories and not agents_data:
            return {"status": "no_data"}

        # 1. 그룹별 보상 비교
        # 그룹 식별: agent_id (OptimalAgent vs SelfishAgent)
        # 현재는 ID로 구분 불가하므로, AgentCell 정보를 받아야 함.
        # 여기서는 가정: player_짝수=Optimal, player_홀수=Selfish
        
        rewards_optimal = []
        rewards_selfish = []
        
        for agent_id, total_reward in agents_data.items():
            # 임시 로직: 짝수/홀수 구분 (AgentCell과 동일)
            # player_N -> N
            try:
                idx = int(agent_id.split("_")[1])
                if idx % 2 == 0:
                    rewards_optimal.append(total_reward)
                else:
                    rewards_selfish.append(total_reward)
            except:
                pass

        # 2. 통계 검정 (t-test)
        t_stat, p_val = 0.0, 1.0
        if len(rewards_optimal) > 1 and len(rewards_selfish) > 1:
            t_stat, p_val = stats.ttest_ind(rewards_optimal, rewards_selfish, equal_var=False)

        # 3. 협력률 계산
        coop_count = sum(1 for t in trajectories if t.get("action") == 0)
        total_actions = len(trajectories)
        coop_rate = coop_count / total_actions if total_actions > 0 else 0.0

        return {
            "metrics": {
                "avg_reward_optimal": np.mean(rewards_optimal) if rewards_optimal else 0.0,
                "avg_reward_selfish": np.mean(rewards_selfish) if rewards_selfish else 0.0,
                "cooperation_rate": coop_rate
            },
            "hypothesis_test": {
                "method": "Welch's t-test",
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "significant": p_val < self.config.significance_level
            }
        }
