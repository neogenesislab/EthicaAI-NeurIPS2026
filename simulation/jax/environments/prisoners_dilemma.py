"""
JAX-based Prisoner's Dilemma Environment
Compatible with Gymnax/JAX-MARL style.
"""
import jax
import jax.numpy as jnp
from jax import lax
import chex
from flax import struct
from typing import Tuple, Dict, Any

@struct.dataclass
class EnvParams:
    # Payoff Matrix (T > R > P > S)
    T: float = 5.0
    R: float = 3.0
    P: float = 1.0
    S: float = 0.0
    
    # Sanction Parameters
    enable_sanction: bool = True
    punishment_cost: float = 1.0
    punishment_fine: float = 4.0
    
    max_steps: int = 100

@struct.dataclass
class EnvState:
    step_count: int
    # History for observation (Last actions)
    # Shape: (2,) for 2 agents
    last_actions: chex.Array 

class PrisonersDilemmaJax:
    def __init__(self):
        self.num_agents = 2
        self.agents = ["player_0", "player_1"]

    @property
    def name(self) -> str:
        return "PrisonersDilemma-v1"

    @property
    def num_actions(self) -> int:
        # 0:C, 1:D, 2:C+P, 3:D+P
        return 4 

    def observation_space(self, params: EnvParams):
        # [Opp_C, Opp_D, Opp_P, Init]
        return 4

    def action_space(self, params: EnvParams):
        return 4

    def reset(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[Dict[str, chex.Array], EnvState]:
        # Reset Logic
        state = EnvState(
            step_count=jnp.int32(0),
            last_actions=jnp.full((self.num_agents,), -1, dtype=jnp.int32) # -1: Initial
        )
        return self.get_obs(state, params), state

    def step(self, key: chex.PRNGKey, state: EnvState, actions: Dict[str, chex.Array], params: EnvParams) -> Tuple[Dict[str, chex.Array], EnvState, Dict[str, float], Dict[str, bool], Dict]:
        # action dict -> array
        a0 = actions["player_0"]
        a1 = actions["player_1"]
        
        # Parse actions (Sanction logic)
        # 0=C, 1=D, 2=C+P, 3=D+P
        def parse_act(a):
            # coop: 0 or 2 -> 0, defect: 1 or 3 -> 1
            is_coop = (a == 0) | (a == 2)
            base = jnp.where(is_coop, 0, 1) # cast to int32 implied by array usage
            is_punish = (a == 2) | (a == 3)
            return base, is_punish
            
        base_a0, punish_a0 = parse_act(a0)
        base_a1, punish_a1 = parse_act(a1)
        
        # Calculate Rewards
        # (0,0)->R, (0,1)->S, (1,0)->T, (1,1)->P
        
        # Vectorized lookup using matrix indexing
        payoff_mat_0 = jnp.array([[params.R, params.S], [params.T, params.P]])
        payoff_mat_1 = jnp.array([[params.R, params.T], [params.S, params.P]])
        
        r0 = payoff_mat_0[base_a0, base_a1]
        r1 = payoff_mat_1[base_a0, base_a1]
        
        # Sanction Effects
        cost = params.punishment_cost
        fine = params.punishment_fine
        
        r0 = r0 - jnp.where(punish_a0, cost, 0.0) - jnp.where(punish_a1, fine, 0.0)
        r1 = r1 - jnp.where(punish_a1, cost, 0.0) - jnp.where(punish_a0, fine, 0.0)
        
        # Update State
        step_count = state.step_count + 1
        last_actions = jnp.stack([a0, a1])
        
        next_state = state.replace(
            step_count=step_count,
            last_actions=last_actions
        )
        
        # Done
        done_bool = step_count >= params.max_steps
        dones = {"player_0": done_bool, "player_1": done_bool, "__all__": done_bool}
        
        rewards = {"player_0": r0, "player_1": r1}
        
        # Obs
        obs = self.get_obs(next_state, params)
        
        info = {}
        
        return obs, next_state, rewards, dones, info

    def get_obs(self, state: EnvState, params: EnvParams) -> Dict[str, chex.Array]:
        # Obs: [Opp_C, Opp_D, Opp_P, Init]
        def make_obs(opp_idx):
            last_a = state.last_actions[opp_idx]
            
            # If initial (-1)
            is_init = (last_a == -1)
            
            # Base action (0,1)
            is_coop = (last_a == 0) | (last_a == 2)
            is_defect = (last_a == 1) | (last_a == 3)
            is_punish = (last_a == 2) | (last_a == 3)
            
            obs_vec = jnp.zeros(4)
            obs_vec = obs_vec.at[0].set(jnp.where((is_coop) & (~is_init), 1.0, 0.0))
            obs_vec = obs_vec.at[1].set(jnp.where((is_defect) & (~is_init), 1.0, 0.0))
            obs_vec = obs_vec.at[2].set(jnp.where((is_punish) & (~is_init), 1.0, 0.0))
            obs_vec = obs_vec.at[3].set(jnp.where(is_init, 1.0, 0.0))
            
            return obs_vec

        return {
            "player_0": make_obs(1),
            "player_1": make_obs(0)
        }

