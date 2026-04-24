"""
BaseCell — 모든 셀의 추상 베이스 클래스

설계 원칙 (사용자 지시):
1. execute(input) → output 인터페이스 통일
2. 상태(state) 관리 — 셀 내부 상태를 딕셔너리로 보관
3. 로그(log) 기록 — 모든 실행 이력 추적
4. 각 셀은 Orchestrator를 통해서만 통신 (직접 호출 금지)
5. 나중에 LangGraph로 전환 가능하도록 인터페이스 설계
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ────────────────────────────────────────────
# 셀 실행 기록
# ────────────────────────────────────────────
@dataclass
class CellLog:
    """셀 단위 실행 로그 엔트리."""
    cell_name: str
    timestamp: float
    duration_sec: float
    input_summary: str
    output_summary: str
    status: str  # "성공", "실패", "건너뜀"
    error: Optional[str] = None


# ────────────────────────────────────────────
# 셀 상태 컨테이너
# ────────────────────────────────────────────
@dataclass
class CellState:
    """셀의 내부 상태를 캡슐화하는 컨테이너.

    Attributes:
        data: 셀 고유 상태 데이터 (자유 형식)
        is_initialized: 초기화 완료 여부
        execution_count: 총 실행 횟수
    """
    data: Dict[str, Any] = field(default_factory=dict)
    is_initialized: bool = False
    execution_count: int = 0


# ────────────────────────────────────────────
# 추상 베이스 셀
# ────────────────────────────────────────────
class BaseCell(ABC):
    """모든 셀의 추상 베이스 클래스.

    하위 클래스는 반드시 다음을 구현해야 한다:
        - name (property): 셀 고유 이름
        - _execute(input_data): 실제 로직

    LangGraph 전환 고려사항:
        - execute()의 시그니처가 LangGraph 노드의
          (state) -> state 패턴과 호환되도록 설계됨
        - state 딕셔너리를 통해 셀 간 데이터를 전달

    사용 예시:
        class MyCell(BaseCell):
            @property
            def name(self) -> str:
                return "MyCell"

            def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                return {"result": input_data["value"] * 2}
    """

    def __init__(self) -> None:
        self._state = CellState()
        self._logs: List[CellLog] = []
        self._logger = logging.getLogger(f"EthicaAI.{self.name}")

    # ── 추상 인터페이스 ──────────────────────
    @property
    @abstractmethod
    def name(self) -> str:
        """셀 고유 이름 (예: 'EnvCell', 'AgentCell')."""
        ...

    @abstractmethod
    def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """실제 비즈니스 로직. 하위 클래스에서 구현.

        Args:
            input_data: Orchestrator로부터 전달받은 입력 데이터.

        Returns:
            출력 데이터 딕셔너리. 다음 셀의 입력이 됨.
        """
        ...

    # ── 공개 API ─────────────────────────────
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """셀 실행 — 로깅/타이밍/에러 처리를 래핑.

        Orchestrator가 호출하는 유일한 진입점.
        직접 _execute()를 호출하지 말 것.

        Args:
            input_data: 입력 데이터 딕셔너리.

        Returns:
            출력 데이터 딕셔너리.

        Raises:
            Exception: _execute 내부에서 발생한 모든 예외를 로그에 기록 후 재발생.
        """
        start = time.time()
        self._logger.info(f"[{self.name}] 실행 시작")
        self._state.execution_count += 1

        try:
            output = self._execute(input_data)
            duration = time.time() - start

            log_entry = CellLog(
                cell_name=self.name,
                timestamp=start,
                duration_sec=round(duration, 4),
                input_summary=self._summarize(input_data),
                output_summary=self._summarize(output),
                status="성공",
            )
            self._logs.append(log_entry)
            self._logger.info(
                f"[{self.name}] 실행 완료 ({duration:.2f}초)"
            )
            return output

        except Exception as e:
            duration = time.time() - start
            log_entry = CellLog(
                cell_name=self.name,
                timestamp=start,
                duration_sec=round(duration, 4),
                input_summary=self._summarize(input_data),
                output_summary="",
                status="실패",
                error=str(e),
            )
            self._logs.append(log_entry)
            self._logger.error(f"[{self.name}] 실행 실패: {e}")
            raise

    def initialize(self, config: Dict[str, Any]) -> None:
        """셀 초기화 (선택 사항, 하위 클래스에서 오버라이드 가능).

        Args:
            config: 초기화에 필요한 설정 딕셔너리.
        """
        self._state.data.update(config)
        self._state.is_initialized = True
        self._logger.info(f"[{self.name}] 초기화 완료")

    def reset(self) -> None:
        """셀 상태 초기화."""
        self._state = CellState()
        self._logger.info(f"[{self.name}] 상태 리셋")

    # ── 상태/로그 접근자 ─────────────────────
    @property
    def state(self) -> CellState:
        """현재 셀 상태를 반환."""
        return self._state

    @property
    def logs(self) -> List[CellLog]:
        """실행 로그 목록을 반환."""
        return self._logs.copy()

    @property
    def last_log(self) -> Optional[CellLog]:
        """가장 최근 실행 로그를 반환."""
        return self._logs[-1] if self._logs else None

    # ── 내부 유틸리티 ────────────────────────
    @staticmethod
    def _summarize(data: Dict[str, Any], max_len: int = 100) -> str:
        """데이터 딕셔너리의 키 목록을 요약 문자열로 변환.

        Args:
            data: 요약할 딕셔너리.
            max_len: 최대 문자열 길이.

        Returns:
            요약 문자열 (예: "keys=[env, agents, config]").
        """
        if not data:
            return "빈 딕셔너리"
        keys_str = ", ".join(data.keys())
        summary = f"keys=[{keys_str}]"
        return summary[:max_len] + "..." if len(summary) > max_len else summary

    def __repr__(self) -> str:
        return (
            f"<{self.name} | "
            f"초기화={'완료' if self._state.is_initialized else '미완'} | "
            f"실행횟수={self._state.execution_count}>"
        )
