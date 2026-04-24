"""
EnvCell 단위 테스트

- EnvCell 초기화 및 환경 생성 테스트
- reset() 기능 테스트
- step() 실행 및 보상/관측 반환 테스트
- AEC(Agent Environment Cycle) 순서 준수 확인
"""
import unittest
from typing import Dict, Any

from simulation.cells.env_cell import EnvCell
from simulation.config import PrisonersDilemmaConfig


class TestEnvCell(unittest.TestCase):
    def setUp(self):
        self.cell = EnvCell()
        # 테스트용 설정 (짧은 라운드)
        self.config = PrisonersDilemmaConfig(num_rounds=10)
        self.cell.initialize({"env_config": self.config})

    def test_initialization(self):
        """환경 초기화 확인."""
        self.assertEqual(self.cell.name, "EnvCell")
        self.assertTrue(self.cell.state.is_initialized)
        self.assertIsNotNone(self.cell.env)
        self.assertEqual(self.cell.state.data["env_name"], "prisoners_dilemma_v1")

    def test_reset(self):
        """리셋 동작 확인."""
        input_data = {"reset": True}
        output = self.cell.execute(input_data)
        
        # 리셋 시 모든 에이전트 관측 반환
        self.assertIn("observations", output)
        self.assertIn("player_0", output["observations"])
        self.assertIn("player_1", output["observations"])
        self.assertEqual(output["observations"]["player_0"][2], 1.0) # 초기 상태 (None)

    def test_step_sequence(self):
        """AEC 스텝 실행 순서 확인."""
        # 1. 리셋
        self.cell.execute({"reset": True})
        
        # 첫 번째 에이전트 (player_0 가정)
        first_agent = self.cell.env.agent_selection
        
        # 2. 첫 번째 행동 실행
        # 0: 협력(C)
        action_input = {"actions": {first_agent: 0}}
        output = self.cell.execute(action_input)
        
        # 다음 에이전트 확인
        next_agent = output["next_agent"]
        self.assertNotEqual(first_agent, next_agent)
        
        # 두 번째 에이전트 행동 실행 (player_1)
        # 1: 배신(D)
        action_input_2 = {"actions": {next_agent: 1}}
        output_2 = self.cell.execute(action_input_2)
        
        # 라운드 종료 후 보상 확인 (player_0: C, player_1: D -> CD)
        # CD 보상: player_0=0, player_1=5
        rewards = output_2["rewards"]
        # 주의: AEC에서는 보상이 다음 step 호출 시 누적되어 반환될 수 있음.
        # EnvCell 구현 상 step 직후 rewards를 반환하므로 즉시 확인 가능해야 함.
        # 단, player_0의 보상은 player_1이 행동한 직후(라운드 끝)에 계산됨.
        
        # player_1이 행동한 시점에 라운드가 끝났다면 보상이 있어야 함
        self.assertEqual(rewards["player_0"], 0.0)
        self.assertEqual(rewards["player_1"], 5.0)

    def test_close(self):
        """환경 종료 테스트."""
        self.cell.close()


if __name__ == "__main__":
    unittest.main()
