"""
PrisonersDilemmaEnv — 반복 죄수의 딜레마 (Iterated Prisoner's Dilemma)

PettingZoo AEC(Agent Environment Cycle) API 준수.
특징:
    - 2인용 제로섬/협력 게임
    - 관측 노이즈 (Noise) 지원
    - 처벌(Sanction) 메커니즘 지원 (행동 공간 확장)

행동 공간 (Discrete 4 if sanction enabled):
    0: Cooperate (C)
    1: Defect (D)
    2: Cooperate + Punish (C+P)
    3: Defect + Punish (D+P)

관측 공간:
    [상대_C, 상대_D, 상대_P, 초기/노이즈]
"""
import random
import numpy as np
import gymnasium
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
from typing import Dict, List, Optional, Any

from simulation.config import PrisonersDilemmaConfig

class PrisonersDilemmaEnv(AECEnv):
    metadata = {'render.modes': ['human'], "name": "prisoners_dilemma_v1"}

    def __init__(self, config: PrisonersDilemmaConfig = None):
        super().__init__()
        self.config = config or PrisonersDilemmaConfig()
        
        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards_map = self.config.reward_matrix
        
        # 행동 공간 설정
        if self.config.enable_sanction:
            # 0=C, 1=D, 2=C+P, 3=D+P
            self.action_spaces = {agent: spaces.Discrete(4) for agent in self.agents}
            obs_dim = 4 # [Opp_C, Opp_D, Opp_P, Init/Noise]
        else:
            # 0=C, 1=D
            self.action_spaces = {agent: spaces.Discrete(2) for agent in self.agents}
            obs_dim = 3 # [Opp_C, Opp_D, Init/Noise]

        # 관측 공간 설정 (0~1 사이 값)
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32) 
            for agent in self.agents
        }
        
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # 내부 상태
        self.num_rounds = 0
        self.last_actions = {} # {agent: raw_action}
        self.actions = {agent: 0 for agent in self.agents} # 현재 라운드 행동 저장
        
        # 관측값 저장을 위한 버퍼 (관측 노이즈 처리를 위해 저장)
        self.last_obs_vector = {} 

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self.num_rounds = 0
        self.last_actions = {}
        self.actions = {agent: 0 for agent in self.agents}
        self.last_obs_vector = {}
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        self.actions[agent] = action

        if self._agent_selector.is_last():
            # 라운드 종료 -> 보상 계산
            self._calculate_rewards()
            
            self.num_rounds += 1
            if self.num_rounds >= self.config.num_rounds:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            # 첫 번째 에이전트 행동 완료
            pass

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def _calculate_rewards(self):
        p0, p1 = self.agents[0], self.agents[1]
        raw_a0, raw_a1 = self.actions[p0], self.actions[p1]
        
        # 행동 해석
        # Enable Sanction: 0=C, 1=D, 2=C+P, 3=D+P
        # Disable Sanction: 0=C, 1=D
        
        def parse_action(act):
            if not self.config.enable_sanction:
                return (0 if act == 0 else 1), False
            # 0, 2 -> Cooperate (0)
            # 1, 3 -> Defect (1)
            is_coop = (act == 0 or act == 2)
            base_act = 0 if is_coop else 1
            is_punish = (act == 2 or act == 3)
            return base_act, is_punish

        base_a0, punish_a0 = parse_action(raw_a0)
        base_a1, punish_a1 = parse_action(raw_a1)
        
        # 1. PD 기본 보상
        # (0,0)->R, (0,1)->S, (1,0)->T, (1,1)->P
        if base_a0 == 0 and base_a1 == 0:
            r0, r1 = self.rewards_map["R"], self.rewards_map["R"]
        elif base_a0 == 0 and base_a1 == 1:
            r0, r1 = self.rewards_map["S"], self.rewards_map["T"]
        elif base_a0 == 1 and base_a1 == 0:
            r0, r1 = self.rewards_map["T"], self.rewards_map["S"]
        else:
            r0, r1 = self.rewards_map["P"], self.rewards_map["P"]
            
        # 2. 처벌 비용/벌금 적용
        if self.config.enable_sanction:
            if punish_a0:
                r0 -= self.config.punishment_cost
                r1 -= self.config.punishment_fine
            if punish_a1:
                r1 -= self.config.punishment_cost
                r0 -= self.config.punishment_fine
        
        self.rewards[p0] = r0
        self.rewards[p1] = r1

        # 관측용 벡터 생성 (미리 계산)
        self.last_obs_vector[p0] = self._make_obs_vector(base_a0, punish_a0)
        self.last_obs_vector[p1] = self._make_obs_vector(base_a1, punish_a1)

    def _make_obs_vector(self, base_act, is_punish):
        # [C, D, P, Init]
        vec = np.zeros(4 if self.config.enable_sanction else 3, dtype=np.float32)
        if base_act == 0:
            vec[0] = 1.0
        else:
            vec[1] = 1.0
            
        if self.config.enable_sanction and is_punish:
            vec[2] = 1.0
            
        return vec

    def observe(self, agent: str):
        opponent = self.agents[1] if agent == self.agents[0] else self.agents[0]
        
        obs_dim = 4 if self.config.enable_sanction else 3
        
        # 초기 상태
        if opponent not in self.last_obs_vector:
            obs = np.zeros(obs_dim, dtype=np.float32)
            obs[-1] = 1.0 # Init flag
            return obs
            
        true_obs = self.last_obs_vector[opponent]
        
        # 노이즈 적용 (관측 불확실성)
        # 노이즈 발생 시 '알 수 없음(Init flag)' 처리하거나 랜덤
        if self.config.noise_prob > 0 and random.random() < self.config.noise_prob:
            obs = np.zeros(obs_dim, dtype=np.float32)
            obs[-1] = 1.0 # Noise == Unknown
            return obs
            
        return true_obs

    def render(self):
        pass

    def close(self):
        pass
