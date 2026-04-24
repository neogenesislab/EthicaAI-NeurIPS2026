"""
BaseCell 단위 테스트

- BaseCell 추상 클래스 상속 및 구현 테스트
- execute() 메서드의 로깅 및 상태 갱신 확인
- 예외 처리 및 에러 로그 확인
"""
import unittest
from typing import Any, Dict
from simulation.cells.base_cell import BaseCell


class MockCell(BaseCell):
    """테스트용 Mock 셀."""
    
    @property
    def name(self) -> str:
        return "MockCell"

    def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in input_data:
            raise ValueError("Test Error")
        return {"result": input_data.get("value", 0) * 2}


class TestBaseCell(unittest.TestCase):
    def setUp(self):
        self.cell = MockCell()

    def test_initial_state(self):
        """초기 상태 확인."""
        self.assertEqual(self.cell.name, "MockCell")
        self.assertFalse(self.cell.state.is_initialized)
        self.assertEqual(self.cell.state.execution_count, 0)
        self.assertEqual(len(self.cell.logs), 0)

    def test_execute_success(self):
        """정상 실행 테스트."""
        input_data = {"value": 10}
        output = self.cell.execute(input_data)

        # 결과 확인
        self.assertEqual(output["result"], 20)
        
        # 상태 업데이트 확인
        self.assertEqual(self.cell.state.execution_count, 1)
        
        # 로그 확인
        self.assertEqual(len(self.cell.logs), 1)
        log = self.cell.last_log
        self.assertEqual(log.status, "성공")
        self.assertEqual(log.cell_name, "MockCell")
        self.assertIsNone(log.error)

    def test_execute_failure(self):
        """실행 실패 및 예외 처리 테스트."""
        input_data = {"error": True}
        
        with self.assertRaises(ValueError):
            self.cell.execute(input_data)

        # 상태 업데이트 확인 (실행 시도는 카운트됨)
        self.assertEqual(self.cell.state.execution_count, 1)

        # 로그 확인
        self.assertEqual(len(self.cell.logs), 1)
        log = self.cell.last_log
        self.assertEqual(log.status, "실패")
        self.assertEqual(str(log.error), "Test Error")

    def test_initialize_and_reset(self):
        """초기화 및 리셋 테스트."""
        config = {"param": 42}
        self.cell.initialize(config)
        
        self.assertTrue(self.cell.state.is_initialized)
        self.assertEqual(self.cell.state.data["param"], 42)

        self.cell.reset()
        self.assertFalse(self.cell.state.is_initialized)
        self.assertEqual(self.cell.state.data, {})
        self.assertEqual(self.cell.state.execution_count, 0)


if __name__ == "__main__":
    unittest.main()
