"""
JAX-based Harvest Environment (SocialJax)
Tragedy of the Commons with Spatial Dynamics.
"""
import jax
import jax.numpy as jnp
from jax import lax, random
import chex
from flax import struct
from typing import Tuple, Dict
from functools import partial

from simulation.jax.environments.common import (
    AgentState, GridState, 
    ACTION_NOOP, ACTION_FORWARD, ACTION_BACKWARD, 
    ACTION_LEFT, ACTION_RIGHT, ACTION_ROT_LEFT, ACTION_ROT_RIGHT, ACTION_BEAM,
    OBJ_EMPTY, OBJ_WALL, OBJ_APPLE, OBJ_AGENT, OBJ_BEAM,
    process_movement, get_agent_view, move_forward
)

@struct.dataclass
class EnvParams:
    # Dynamics
    spawn_prob_factor: float = 0.05 # P_spawn approx factor * N_neighbors
    # Or specific table: 0->0, 1->0.005, 2->0.02, 3+->0.05 etc.
    # DeepMind paper uses specific regrowth rates based on neighbors.
    
    # Rewards
    reward_apple: float = 1.0
    cost_beam: float = -0.1
    
    # Punishment
    freeze_duration: int = 25
    
    max_steps: int = 1000

@struct.dataclass
class EnvState(GridState):
    key: chex.PRNGKey

class HarvestJax:
    def __init__(self, num_agents=5, height=25, width=18):
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        # Map size can be smaller for Harvest to increase density/conflict?
        # Use Standard provided size.
        self.height = height
        self.width = width

    @property
    def name(self) -> str:
        return "Harvest-v0"

    @property
    def num_actions(self) -> int:
        return 8 

    def observation_space(self, params: EnvParams):
        # 11x11x6 (Channels)
        # Empty, Wall, Apple, Agent, Beam, ?
        return (11, 11, 6)

    def action_space(self, params: EnvParams):
        return 8

    def reset(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[Dict[str, chex.Array], EnvState]:
        h, w = self.height, self.width
        grid = jnp.zeros((h, w), dtype=jnp.int32)
        
        # Walls around
        grid = grid.at[0, :].set(OBJ_WALL)
        grid = grid.at[-1, :].set(OBJ_WALL)
        grid = grid.at[:, 0].set(OBJ_WALL)
        grid = grid.at[:, -1].set(OBJ_WALL)
        
        # Initial Apples?
        # Spawn some apples randomly or fill patterns?
        # Random fill with low probability
        key, spawn_key, agent_key = random.split(key, 3)
        initial_apple_prob = 0.2
        spawn_mask = random.uniform(spawn_key, shape=(h, w)) < initial_apple_prob
        
        valid_loc = (grid == OBJ_EMPTY)
        grid = jnp.where(spawn_mask & valid_loc, OBJ_APPLE, grid)
        
        # Agents
        agent_states = AgentState(
            pos=jnp.zeros((self.num_agents, 2), dtype=jnp.int32), # At (0,0) - WALL! Needs fix.
            dir=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            reward=jnp.zeros((self.num_agents,), dtype=jnp.float32),
            frozen=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            last_action=jnp.zeros((self.num_agents,), dtype=jnp.int32)
        )
        
        # Better spawn logic: find empty spots?
        # For simplicity, spawn at valid fixed locations or center.
        # Let's spawn at (1, 1) to (1, N) for now (safe zone).
        
        start_xs = jnp.arange(1, self.num_agents + 1)
        start_ys = jnp.ones(self.num_agents, dtype=jnp.int32)
        agent_states = agent_states.replace(
            pos=jnp.stack([start_ys, start_xs], axis=1)
        )
        
        state = EnvState(
            grid=grid,
            agents=agent_states,
            step_count=0,
            key=key
        )
        
        return self.get_obs(state, params), state

    def step(self, key: chex.PRNGKey, state: EnvState, actions: Dict[str, chex.Array], params: EnvParams) -> Tuple[Dict[str, chex.Array], EnvState, Dict[str, float], Dict[str, bool], Dict]:
        # 1. Process Freeze Decrement
        # If frozen > 0, decrement.
        # If frozen, action is forced NOOP.
        
        frozen_mask = (state.agents.frozen > 0)
        
        # Override Actions for frozen agents
        action_arr = jnp.stack([actions[a] for a in self.agents])
        action_arr = jnp.where(frozen_mask, ACTION_NOOP, action_arr)
        
        new_frozen = jnp.maximum(state.agents.frozen - 1, 0)
        
        # 2. Process Actions (Move)
        process_move_vmap = jax.vmap(process_movement, in_axes=(0, 0, None, None, None))
        new_agents = process_move_vmap(state.agents, action_arr, state.grid, self.height, self.width)
        
        # Maintain frozen status in new_agents (process_movement returns new state struct, need to inject frozen)
        new_agents = new_agents.replace(frozen=new_frozen)
        
        # 3. Process Beam (Punishment)
        # In Harvest, Beam freezes other agents.
        # Ray trace? or 1-cell logic? 
        # Standard: multi-cell beam. Let's stick to 1-cell front for simplicity first.
        # OR: Standard Harvest beam is wide and long. 
        # Simpler: 1 cell front beam that freezes if agent is there.
        
        is_beam = (action_arr == ACTION_BEAM)
        
        # Identify target positions of beams
        def get_target(agent):
            return move_forward(agent.pos, agent.dir)
            
        target_pos = jax.vmap(get_target)(state.agents) # (N, 2)
        
        # Check if any agent is at target_pos
        # Matrix (N_beamers, N_targets)
        # If Beamer i targets pos P, and Agent j is at pos P, Agent j is Valid Hit.
        
        agent_pos_matrix = new_agents.pos # (N, 2)
        
        # Vectorized check: all vs all
        # hit_matrix[i, j] = True if agent i hits agent j
        # condition: is_beam[i] AND target_pos[i] == agent_pos[j]
        
        # Expand dims for digest
        # target_pos: (N, 1, 2)
        # agent_pos: (1, N, 2)
        # equality: (N, N, 2) -> all(axis=2) -> (N, N)
        
        hits = (target_pos[:, None, :] == agent_pos_matrix[None, :, :]).all(axis=2)
        hits = hits & is_beam[:, None]
        
        # Exclude self-hit? (Can't be in front of self usually)
        
        # Who gets frozen?
        # Any column j that has True means agent j was hit.
        is_hit = jnp.any(hits, axis=0) # (N,)
        
        # Apply Freeze
        # If hit, set frozen to 25.
        final_frozen = jnp.where(is_hit, params.freeze_duration, new_agents.frozen)
        new_agents = new_agents.replace(frozen=final_frozen)
        
        # 4. Process Consumption (Eat Apple)
        rewards_arr = jnp.zeros(self.num_agents)
        
        # Mask: Agents NOT frozen can eat
        can_eat = (final_frozen == 0)
        
        # Identify apples at agent positions
        agent_pos_idx = new_agents.pos
        # Boolean mask of grid
        # Handle contention logic
        
        def check_and_eat_reward(agent_pos, grid, active_mask):
             # Only if active
             is_apple = (grid[agent_pos[0], agent_pos[1]] == OBJ_APPLE)
             return jnp.where(is_apple & active_mask, params.reward_apple, 0.0)
             
        rewards_apple = jax.vmap(check_and_eat_reward, in_axes=(0, None, 0))(
            agent_pos_idx, state.grid, can_eat
        )
        
        # Remove Apples
        # Only remove if someone ate it.
        # Construct mask of Eaten Cells
        eaten_cells_mask = jnp.zeros_like(state.grid, dtype=jnp.bool_)
        eaten_cells_mask = eaten_cells_mask.at[agent_pos_idx[:,0], agent_pos_idx[:,1]].set(can_eat) 
        # Note: If multiple agents at same cell, both "eat", but cell cleared once.
        # But if cell didn't have apple?
        is_apple_grid = (state.grid == OBJ_APPLE)
        actually_eaten = eaten_cells_mask & is_apple_grid
        
        new_grid_mid = jnp.where(actually_eaten, OBJ_EMPTY, state.grid)
        
        # Beaming Cost
        rewards_cost = jnp.where(is_beam, params.cost_beam, 0.0)
        total_rewards = rewards_apple + rewards_cost
        
        # 5. Regrowth Dynamics
        # For every empty cell, check neighbors.
        # Convolution with kernel [3x3]?
        # L1 distance <= 2 -> Diamond shape kernel.
        # Kernel 5x5 for L1=2
        # Center 1? No, neighbors.
        
        # Kernel for L1 <= 2:
        # 0 0 1 0 0
        # 0 1 1 1 0
        # 1 1 0 1 1  (Center is 0, we count neighbors)
        # 0 1 1 1 0
        # 0 0 1 0 0
        
        kernel = jnp.array([
            [0,0,1,0,0],
            [0,1,1,1,0],
            [1,1,0,1,1],
            [0,1,1,1,0],
            [0,0,1,0,0]
        ], dtype=jnp.float32)
        
        # Apple Mask (1.0 where apple)
        apple_map = (new_grid_mid == OBJ_APPLE).astype(jnp.float32)
        
        # Convolve
        # mode='same'
        neighbor_counts = jax.scipy.signal.convolve2d(apple_map, kernel, mode='same')
        
        # Spawn Prob
        # Simple Logic: P = neighbors * factor
        spawn_probs = neighbor_counts * params.spawn_prob_factor
        # Cap at 1.0?
        
        key, spawn_key = random.split(key)
        spawn_execution = random.uniform(spawn_key, shape=new_grid_mid.shape) < spawn_probs
        
        valid_growth = (new_grid_mid == OBJ_EMPTY)
        new_grid = jnp.where(spawn_execution & valid_growth, OBJ_APPLE, new_grid_mid)
        
        # Update State
        new_state = state.replace(
            grid=new_grid,
            agents=new_agents,
            step_count=state.step_count + 1,
            key=key
        )
        
        # Obs
        obs = self.get_obs(new_state, params)
        
        # Standardize Output
        rewards_dict = {a: total_rewards[i] for i, a in enumerate(self.agents)}
        dones = {"__all__": state.step_count >= params.max_steps}
        for a in self.agents:
            dones[a] = dones["__all__"]
            
        info = {
            "apples_eaten": jnp.sum(actually_eaten),
            "frozen_agents": jnp.sum(is_hit)
        }
        
        return obs, new_state, rewards_dict, dones, info

    def get_obs(self, state: EnvState, params: EnvParams) -> Dict[str, chex.Array]:
        # Implementation similar to Cleanup
        display_grid = state.grid
        agent_pos_idx = state.agents.pos
        agent_mask = jnp.zeros_like(display_grid, dtype=jnp.bool_)
        agent_mask = agent_mask.at[agent_pos_idx[:, 0], agent_pos_idx[:, 1]].set(True)
        display_grid = jnp.where(agent_mask, OBJ_AGENT, display_grid)
        
        from simulation.jax.environments.common import get_agent_view 
        
        def _get_agent_obs(agent):
            raw_view = get_agent_view(display_grid, agent.pos, agent.dir, view_size=11)
            obs = jax.nn.one_hot(raw_view, num_classes=6) 
            return obs
            
        obs_batch = jax.vmap(_get_agent_obs)(state.agents)
        return {a: obs_batch[i] for i, a in enumerate(self.agents)}
