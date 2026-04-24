"""
AgentCell — 에이전트 관리 셀

역할:
    - 다수의 에이전트 인스턴스 생성 및 관리
    - 관측(Observation) 입력 → 행동(Action) 출력 결정
    - EnvCell과 주고받는 데이터 중계

설계:
    - AgentConfig에 따라 Selfish, Optimal, Bounded 비율대로 생성
    - step() 호출 시 현재 턴인 agent에게 act() 요청
"""
from typing import Any, Dict, List

from simulation.cells.base_cell import BaseCell
from simulation.config import AgentConfig
from simulation.agents.base_agent import BaseAgent
from simulation.agents.selfish import SelfishAgent
from simulation.agents.optimal import OptimalAgent
from simulation.agents.bounded import BoundedAgent


class AgentCell(BaseCell):
    """에이전트 집단 관리 셀."""
    
    def __init__(self):
        super().__init__()
        self.agents: Dict[str, BaseAgent] = {}
        self.config: AgentConfig = None

    @property
    def name(self) -> str:
        return "AgentCell"

    def initialize(self, config: Dict[str, Any]) -> None:
        """에이전트 인구 생성."""
        super().initialize(config)
        
        # config 로드 (dataclass 주입 가정)
        self.config = config.get("agent_config") or AgentConfig()
        
        # 에이전트 생성
        # 예: Selfish=40, Optimal=40, Bounded=20 (총 100명 가정 시)
        # 하지만 EnvCell은 기본적으로 player_0, player_1 두 명만 요구 (현재 prisoners_dilemma 구현 상)
        # 따라서 여기서는 매핑 전략이 필요함.
        # 전략: EnvCell의 agent_id ("player_n")에 실제 Agent 인스턴스를 할당.
        # 여기서는 테스트용으로 2명만 생성하되, 타입을 섞어서 배치.
        
        env_agents = config.get("env_agents", ["player_0", "player_1"])
        
        self.agents = {}
        for idx, agent_id in enumerate(env_agents):
            # 간단한 로직: 짝수=Optimal, 홀수=Selfish
            if idx % 2 == 0:
                self.agents[agent_id] = OptimalAgent(agent_id, config)
                self._logger.info(f"{agent_id} -> OptimalAgent 생성")
            else:
                self.agents[agent_id] = SelfishAgent(agent_id, config)
                self._logger.info(f"{agent_id} -> SelfishAgent 생성")
                
        self.state.data["num_agents"] = len(self.agents)

    def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """행동 결정 실행.

        Args:
            input_data: {
                "observations": {agent_id: obs, ...},
                "next_agent": agent_id (AEC 환경용)
            }

        Returns:
            {
                "actions": {agent_id: action, ...}
            }
        """
        observations = input_data.get("observations", {})
        next_agent = input_data.get("next_agent")
        
        actions = {}
        
        # AEC 모드: 특정 에이전트만 행동
        if next_agent and next_agent in self.agents:
            obs = observations.get(next_agent)
            if obs is not None:
                agent = self.agents[next_agent]
                action = agent.act(obs)
                actions[next_agent] = action
        
        # 병렬 모드 (모든 관측에 대해 행동)
        elif observations:
            for agent_id, obs in observations.items():
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    action = agent.act(obs)
                    actions[agent_id] = action

        return {"actions": actions}
