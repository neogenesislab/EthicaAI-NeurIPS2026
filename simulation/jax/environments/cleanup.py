"""
JAX-based Cleanup Environment (SocialJax)
Public Goods Game with Spatial & Temporal Dynamics.
"""
import jax
import jax.numpy as jnp
from jax import lax, random
import chex
from flax import struct
from typing import Tuple, Dict

from simulation.jax.environments.common import (
    AgentState, GridState, 
    ACTION_NOOP, ACTION_FORWARD, ACTION_BACKWARD, 
    ACTION_LEFT, ACTION_RIGHT, ACTION_ROT_LEFT, ACTION_ROT_RIGHT, ACTION_BEAM,
    OBJ_EMPTY, OBJ_WALL, OBJ_APPLE, OBJ_WASTE, OBJ_AGENT, OBJ_BEAM,
    process_movement, get_agent_view
)

@struct.dataclass
class EnvParams:
    # Dynamics
    apple_respawn_prob: float = 0.05
    waste_spawn_prob: float = 0.5
    threshold_depletion: float = 0.4
    threshold_restoration: float = 0.0
    
    # Rewards
    reward_apple: float = 1.0
    cost_beam: float = -0.1
    
    max_steps: int = 1000
    
    # Pheromone Dynamics
    pheromone_rate: float = 0.1 # Diffusion
    pheromone_decay: float = 0.01 # Evaporation
    pheromone_deposit: float = 1.0 # Amount deposited per step by agent

@struct.dataclass
class EnvState(GridState):
    # Inherits grid, agents, step_count
    pheromone_grid: chex.Array # (H, W)
    key: chex.PRNGKey

class CleanupJax:
    def __init__(self, num_agents=5, height=25, width=18):
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.height = height
        self.width = width

    @property
    def name(self) -> str:
        return "Cleanup-v0"

    @property
    def num_actions(self) -> int:
        return 8 

    def observation_space(self, params: EnvParams):
        # 11x11x7 (Channels) - Added Pheromone
        return (11, 11, 7)

    def action_space(self, params: EnvParams):
        return 8

    def reset(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[Dict[str, chex.Array], EnvState]:
        """환경 초기화: 그리드 생성, Waste 배치, 에이전트 랜덤 배치.
        
        레이아웃:
        - 행 0: 벽 (Wall)
        - 행 1 ~ (h//2 - 1): 과수원 (Orchard) — 에이전트 스폰 영역
        - 행 h//2 ~ (h - 2): 강 (River) — Waste로 초기 충전
        - 행 h-1: 벽 (Wall)
        - 열 0, w-1: 벽 (Wall)
        """
        h, w = self.height, self.width
        grid = jnp.zeros((h, w), dtype=jnp.int32)
        
        # 1. 벽(Wall) 배치 — 외곽 테두리
        grid = grid.at[0, :].set(OBJ_WALL)
        grid = grid.at[-1, :].set(OBJ_WALL)
        grid = grid.at[:, 0].set(OBJ_WALL)
        grid = grid.at[:, -1].set(OBJ_WALL)
        
        # 2. 강(River) 영역에 Waste 초기 배치 — 딜레마 구조의 핵심
        # River: row h//2 ~ h-2, col 1 ~ w-2 (벽 제외)
        row_indices = jnp.arange(h)
        col_indices = jnp.arange(w)
        river_mask = (
            (row_indices[:, None] >= h // 2) &
            (row_indices[:, None] < h - 1) &
            (col_indices[None, :] >= 1) &
            (col_indices[None, :] < w - 1)
        )
        grid = jnp.where(river_mask, OBJ_WASTE, grid)
        
        # 3. 에이전트 랜덤 배치 — 과수원(Orchard) 영역 내
        # Orchard: row 1 ~ (h//2 - 1), col 1 ~ (w - 2)
        orchard_rows = jnp.arange(1, h // 2)       # 내부 행 (벽 제외)
        orchard_cols = jnp.arange(1, w - 1)         # 내부 열 (벽 제외)
        # 가능한 모든 위치를 2D 그리드로 생성
        row_grid, col_grid = jnp.meshgrid(orchard_rows, orchard_cols, indexing='ij')
        all_positions = jnp.stack([row_grid.ravel(), col_grid.ravel()], axis=-1)  # (N_valid, 2)
        
        # 랜덤 셔플 후 앞에서 num_agents개 선택
        key, pos_key, dir_key = random.split(key, 3)
        num_valid = all_positions.shape[0]
        perm = random.permutation(pos_key, num_valid)
        agent_positions = all_positions[perm[:self.num_agents]]  # (num_agents, 2)
        
        # 방향도 랜덤 (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        agent_dirs = random.randint(dir_key, (self.num_agents,), 0, 4)
        
        agent_states = AgentState(
            pos=agent_positions,
            dir=agent_dirs,
            reward=jnp.zeros((self.num_agents,), dtype=jnp.float32),
            frozen=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            last_action=jnp.zeros((self.num_agents,), dtype=jnp.int32)
        )
        
        state = EnvState(
            grid=grid,
            agents=agent_states,
            pheromone_grid=jnp.zeros((h, w), dtype=jnp.float32),
            step_count=0,
            key=key
        )
        
        return self.get_obs(state, params), state

    def step(self, key: chex.PRNGKey, state: EnvState, actions: Dict[str, chex.Array], params: EnvParams) -> Tuple[Dict[str, chex.Array], EnvState, Dict[str, float], Dict[str, bool], Dict]:
        # 1. Process Actions (Move & Beam)
        
        # Convert dict actions to array (Order matters: agent_0..agent_N)
        # Assuming sorted keys
        action_arr = jnp.stack([actions[a] for a in self.agents])
        
        # vmap process_movement over agents
        process_move_vmap = jax.vmap(process_movement, in_axes=(0, 0, None, None, None))
        
        # Note: Simultaneous move (no collision check between agents for V1, just walls)
        new_agents = process_move_vmap(state.agents, action_arr, state.grid, self.height, self.width)
        
        # 2. Process Beam (Cleaning)
        # If action == BEAM, clear waste in front.
        is_beam = (action_arr == ACTION_BEAM)
        
        # Function to get beam mask for single agent
        def get_beam_mask(agent):
            # 1 cell in front
            from simulation.jax.environments.common import move_forward # Import here to ensure scope
            target_pos = move_forward(agent.pos, agent.dir) 
            
            # Check bounds
            valid = (target_pos[0] >= 0) & (target_pos[0] < self.height) & \
                    (target_pos[1] >= 0) & (target_pos[1] < self.width)
            
            mask = jnp.zeros((self.height, self.width), dtype=jnp.bool_)
            mask = mask.at[target_pos[0], target_pos[1]].set(valid)
            return mask
            
        beam_masks = jax.vmap(get_beam_mask)(state.agents) # (N, H, W)
        # Filter by is_beam action
        beam_active = beam_masks * is_beam[:, None, None]
        total_beam_mask = jnp.any(beam_active, axis=0)
        
        # Update Grid: Remove Waste where Beam hits
        # Waste is OBJ_WASTE (3)
        grid_has_waste = (state.grid == OBJ_WASTE)
        grid_cleaned = jnp.where(total_beam_mask & grid_has_waste, OBJ_EMPTY, state.grid)
        
        # Count cleaned for stats
        cleaned_count = jnp.sum(total_beam_mask & grid_has_waste)
        
        # 2.5 Pheromone Dynamics
        # Deposit: Agents emit pheromone at their current position
        # For efficiency, scatter add
        # agent pos (N, 2)
        
        # Create deposit grid
        deposit_grid = jnp.zeros_like(state.pheromone_grid)
        deposit_grid = deposit_grid.at[new_agents.pos[:, 0], new_agents.pos[:, 1]].add(params.pheromone_deposit)
        
        # Add to existing
        current_pheromone = state.pheromone_grid + deposit_grid
        
        # Diffuse & Decay
        from simulation.jax.environments.common import diffuse_stencil
        new_pheromone = diffuse_stencil(current_pheromone, params.pheromone_rate, params.pheromone_decay)
        
        # 3. Eat Apples
        # Map agent pos to grid count for contention check (simplified: all get it)
        
        # Rewards
        rewards_arr = jnp.zeros(self.num_agents)
        
        def check_reward(agent):
            y, x = agent.pos
            # Check if apple IS there (in grid_cleaned)
            is_at_apple = (grid_cleaned[y, x] == OBJ_APPLE)
            return jnp.where(is_at_apple, params.reward_apple, 0.0)
            
        reward_apple = jax.vmap(check_reward)(new_agents)
        
        # Remove eaten apples from grid
        # Identify cells with apples AND agents
        agent_pos_idx = new_agents.pos
        # Create a boolean mask of agent positions
        agent_mask = jnp.zeros_like(grid_cleaned, dtype=jnp.bool_)
        agent_mask = agent_mask.at[agent_pos_idx[:, 0], agent_pos_idx[:, 1]].set(True)
        
        is_apple = (grid_cleaned == OBJ_APPLE)
        eaten_mask = is_apple & agent_mask
        
        new_grid_mid = jnp.where(eaten_mask, OBJ_EMPTY, grid_cleaned)
        
        # Cost of Beam
        reward_cost = jnp.where(is_beam, params.cost_beam, 0.0)
        
        total_rewards = reward_apple + reward_cost
        
        # 4. Spawn Dynamics
        key, k1, k2 = random.split(key, 3)
        
        # Waste Density (in River)
        river_mask = (jnp.arange(self.height)[:, None] >= self.height // 2)
        total_waste = jnp.sum((new_grid_mid == OBJ_WASTE) & river_mask)
        river_area = jnp.sum(river_mask)
        waste_density = total_waste / (river_area + 1e-6)
        
        # Apple Rate = f(waste)
        # If waste < threshold, spawn apples
        current_spawn_prob = jnp.where(waste_density < params.threshold_depletion, 
                                     params.apple_respawn_prob, 0.0) # Linear scaling better? Step for now.
        
        # Random Spawn mask
        spawn_mask = random.uniform(k1, shape=new_grid_mid.shape) < current_spawn_prob
        # Only spawn in valid empty spots (Orchard area)
        orchard_mask = (jnp.arange(self.height)[:, None] < self.height // 2)
        valid_spawn = (new_grid_mid == OBJ_EMPTY) & orchard_mask
        
        new_grid = jnp.where(spawn_mask & valid_spawn, OBJ_APPLE, new_grid_mid)
        
        # Waste Spawn
        waste_spawn_mask = random.uniform(k2, shape=new_grid.shape) < params.waste_spawn_prob
        valid_waste = (new_grid == OBJ_EMPTY) & river_mask
        
        new_grid = jnp.where(waste_spawn_mask & valid_waste, OBJ_WASTE, new_grid)
        
        # Update State
        new_state = state.replace(
            grid=new_grid,
            agents=new_agents,
            pheromone_grid=new_pheromone,
            step_count=state.step_count + 1,
            key=key
        )
        
        # Obs
        obs = self.get_obs(new_state, params)
        
        # Outputs
        rewards_dict = {a: total_rewards[i] for i, a in enumerate(self.agents)}
        dones = {"__all__": state.step_count >= params.max_steps}
        for a in self.agents:
            dones[a] = dones["__all__"]
            
        info = {
            "waste_density": waste_density,
            "apples_eaten": jnp.sum(eaten_mask),
            "cleaned_count": cleaned_count
        }
        
        return obs, new_state, rewards_dict, dones, info

    def get_obs(self, state: EnvState, params: EnvParams) -> Dict[str, chex.Array]:
        # Return 11x11x6 View for each agent
        # Channel 0: Empty
        # Channel 1: Wall
        # Channel 2: Apple
        # Channel 3: Waste
        # Channel 4: Agent
        # Channel 5: Beam (Not implemented in grid state yet, maybe strict logic?)
        
        # Grid contains: OBJ_EMPTY(0), OBJ_WALL(1), OBJ_APPLE(2), OBJ_WASTE(3)
        # Agents are separate.
        
        # Construct a Full Grid with Agents
        # Base grid
        display_grid = state.grid
        
        # Overlay Agents (OBJ_AGENT=4)
        agent_pos_idx = state.agents.pos
        # Create a mask of agents
        agent_mask = jnp.zeros_like(display_grid, dtype=jnp.bool_)
        agent_mask = agent_mask.at[agent_pos_idx[:, 0], agent_pos_idx[:, 1]].set(True)
        
        display_grid = jnp.where(agent_mask, OBJ_AGENT, display_grid)
        
        # vmap get_view
        from simulation.jax.environments.common import get_agent_view 
        
        def _get_agent_obs(agent):
            # 1. Get Integer View (11x11)
            raw_view = get_agent_view(display_grid, agent.pos, agent.dir, view_size=11)
            
            # 2. One-Hot Encode (Channels 0-5)
            # Channels: Empty, Wall, Apple, Waste, Agent, Beam
            # Current raw_view has 0, 1, 2, 3, 4. (Beam 5 not in map yet)
            
            obs_onehot = jax.nn.one_hot(raw_view, num_classes=6) # (11, 11, 6)
            
            # 3. Add Pheromone Channel (Channel 6)
            # Get Pheromone View
            p_view = get_agent_view(state.pheromone_grid, agent.pos, agent.dir, view_size=11)
            # p_view is (11, 11) float
            
            # Concatenate
            obs = jnp.concatenate([obs_onehot, p_view[..., None]], axis=-1) # (11, 11, 7)
            return obs
            
        # vmap over agents
        obs_batch = jax.vmap(_get_agent_obs)(state.agents) # (N, 11, 11, 6)
        
        # Convert to Dict
        return {a: obs_batch[i] for i, a in enumerate(self.agents)}
