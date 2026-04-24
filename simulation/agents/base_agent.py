"""
BaseAgent — 모든 에이전트의 추상 베이스 클래스

보고서 §2.1~2.2의 효용 함수 구조를 반영할 수 있는 인터페이스 제공.
하위 클래스(Selfish, Optimal, Bounded)는 act() 메서드를 통해 행동 결정.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseAgent(ABC):
    """에이전트 추상 클래스."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.total_reward = 0.0

    @abstractmethod
    def act(self, observation: np.ndarray) -> int:
        """관측을 바탕으로 행동 선택.

        Args:
            observation: 환경으로부터 받은 관측 벡터 (numpy array)

        Returns:
            action: 선택한 행동 (int)
        """
        pass

    def update(self, reward: float, info: Dict[str, Any] = None):
        """보상 및 정보 업데이트 (학습용).

        Args:
            reward: 받은 보상
            info: 추가 정보 (메타 보상 계산용 등)
        """
        self.total_reward += reward

    def reset(self):
        """에이전트 상태 초기화."""
        self.total_reward = 0.0
