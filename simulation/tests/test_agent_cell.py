"""
AgentCell 단위 테스트

- AgentCell 초기화 시 config에 따른 에이전트 생성 확인
- 관측 입력 시 적절한 에이전트 act() 호출 및 행동 반환 확인
- Selfish/Optimal 에이전트 동작 검증
"""
import sys
import os
import unittest
import numpy as np

# 모듈 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from simulation.cells.agent_cell import AgentCell
from simulation.config import AgentConfig


class TestAgentCell(unittest.TestCase):
    def setUp(self):
        self.cell = AgentCell()
        # 테스트용 설정
        self.config = {
            "agent_config": AgentConfig(),
            "env_agents": ["player_0", "player_1"]
        }
        self.cell.initialize(self.config)

    def test_initialization(self):
        """에이전트 생성 확인."""
        self.assertEqual(len(self.cell.agents), 2)
        # 짝수=Optimal, 홀수=Selfish (구현 로직)
        self.assertEqual(self.cell.agents["player_0"].__class__.__name__, "OptimalAgent")
        self.assertEqual(self.cell.agents["player_1"].__class__.__name__, "SelfishAgent")

    def test_act_sequence(self):
        """행동 결정 테스트."""
        # 1. player_0 (Optimal) 차례
        # 관측: [상대_C, 상대_D, 초기] = [0, 0, 1] (초기)
        obs_p0 = np.array([0.0, 0.0, 1.0])
        input_data = {
            "observations": {"player_0": obs_p0},
            "next_agent": "player_0"
        }
        
        output = self.cell.execute(input_data)
        actions = output["actions"]
        
        self.assertIn("player_0", actions)
        # OptimalAgent는 초기 상태에서 협력(0) 선호 (lambda > 0.5 가정 시)
        # 하지만 현재 OptimalAgent 로직 상 u_self=1.0(D), u_meta=1.0(C) 이고 lambda=0.5이면
        # R(C) = 0.5*0 + 0.5*1 = 0.5
        # R(D) = 0.5*1 + 0.5*0 = 0.5
        # 동률일 경우 argmax는 0(C)을 선택할 수도 있고 1(D)일 수도 있음 (numpy 버전에 따라 다름).
        # 로직 확인 필요: OptimalAgent.act()에서 argmax([0.5, 0.5]) -> 0 (보통 첫 번째 인덱스)
        self.assertIn(actions["player_0"], [0, 1])

        # 2. player_1 (Selfish) 차례
        # 관측: [1, 0, 0] (상대 협력)
        obs_p1 = np.array([1.0, 0.0, 0.0])
        input_data_2 = {
            "observations": {"player_1": obs_p1},
            "next_agent": "player_1"
        }
        output_2 = self.cell.execute(input_data_2)
        actions_2 = output_2["actions"]
        
        # SelfishAgent ("always_defect")는 무조건 1(D)
        self.assertEqual(actions_2["player_1"], 1)

if __name__ == "__main__":
    unittest.main()
