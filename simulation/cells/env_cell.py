"""
EnvCell — 게임이론 환경 셀

역할:
    - PettingZoo/Gymnasium 환경 생명주기 관리 (reset, step, close)
    - 에이전트 행동(Action) 입력 → 관측(Observation), 보상(Reward) 출력
    - 다중 환경 병렬 처리 (VectorEnv) 준비

입력(Input):
    - actions: Dict[agent_id, action]
    - reset: bool (True시 환경 재설정)

출력(Output):
    - observations: Dict[agent_id, obs]
    - rewards: Dict[agent_id, reward]
    - terminations: Dict[agent_id, bool]
    - truncations: Dict[agent_id, bool]
    - infos: Dict[agent_id, info]
"""
from typing import Any, Dict, Optional

from simulation.cells.base_cell import BaseCell
from simulation.config import PrisonersDilemmaConfig
from simulation.environments.prisoners_dilemma import PrisonersDilemmaEnv


class EnvCell(BaseCell):
    """환경 관리 셀 (죄수의 딜레마 등)."""
    
    def __init__(self):
        super().__init__()
        self.env: Optional[PrisonersDilemmaEnv] = None
        self.config: Optional[PrisonersDilemmaConfig] = None

    @property
    def name(self) -> str:
        return "EnvCell"

    def initialize(self, config: Dict[str, Any]) -> None:
        """환경 설정 및 생성."""
        super().initialize(config)
        
        # 설정 로드
        # config 딕셔너리에서 dataclass로 변환 가정 (여기서는 직접 인스턴스 주입)
        self.config = config.get("env_config") or PrisonersDilemmaConfig()
        
        # 환경 생성
        self.env = PrisonersDilemmaEnv(config=self.config)
        self.env.reset()
        
        self.state.data["env_name"] = self.env.metadata["name"]
        self._logger.info(f"체크포인트: 환경 {self.state.data['env_name']} 생성 완료")

    def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """환경 스텝 실행.

        Args:
            input_data: {
                "actions": {agent_id: action, ...},
                "reset": bool (optional)
            }

        Returns:
            {
                "observations": ...,
                "rewards": ...,
                "terminations": ...,
                "truncations": ...,
                "infos": ...
            }
        """
        if not self.env:
            raise RuntimeError("환경이 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")

        # 리셋 요청 처리
        if input_data.get("reset", False):
            self.env.reset()
            obs = {a: self.env.observe(a) for a in self.env.agents}
            return {
                "observations": obs,
                "rewards": {a: 0.0 for a in self.env.agents},
                "terminations": {a: False for a in self.env.agents},
                "truncations": {a: False for a in self.env.agents},
                "infos": {a: {} for a in self.env.agents},
                "status": "reset_complete"
            }

        # 행동 실행
        actions = input_data.get("actions", {})
        if not actions:
            # 초기 상태 관측 반환 (또는 대기)
             obs = {a: self.env.observe(a) for a in self.env.agents}
             return {"observations": obs, "status": "waiting_for_actions"}

        # 각 에이전트 행동 적용
        # AEC 환경은 agent_iter 순서대로 step() 호출 필요
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        # 현재 턴인 에이전트 확인
        current_agent = self.env.agent_selection
        action = actions.get(current_agent)
        
        if action is not None:
            self.env.step(action)
            
            # 스텝 후 상태 수집
            rewards[current_agent] = self.env.rewards[current_agent]
            terminations[current_agent] = self.env.terminations[current_agent]
            truncations[current_agent] = self.env.truncations[current_agent]
            infos[current_agent] = self.env.infos[current_agent]
        else:
            self._logger.warning(f"에이전트 {current_agent}의 행동이 없습니다.")

        # 다음 관측
        next_agent = self.env.agent_selection
        observations = {next_agent: self.env.observe(next_agent)}

        return {
            "observations": observations, # 다음 행동할 에이전트의 관측만 반환 (AEC)
            "rewards": rewards,
            "terminations": terminations,
            "truncations": truncations,
            "infos": infos,
            "next_agent": next_agent
        }

    def close(self):
        if self.env:
            self.env.close()
