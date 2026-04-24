"""
Common Grid Engine for SocialJax (Cleanup/Harvest)
Implements core mechanics: Movement, Rotation, Partial Observation, Beams.
"""
import jax
import jax.numpy as jnp
from jax import lax
import chex
from flax import struct
from typing import Tuple, Dict, Any

# --- Constants ---
# Orientation
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Actions
# 0: No-op, 1: Step Forward, 2: Step Backward, 3: Step Left, 4: Step Right
# 5: Rotate Left, 6: Rotate Right, 7: Beam (Interact)
ACTION_NOOP = 0
ACTION_FORWARD = 1
ACTION_BACKWARD = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_ROT_LEFT = 5
ACTION_ROT_RIGHT = 6
ACTION_BEAM = 7

# Map Objects (Channels or IDs)
# We use One-Hot Channels for Observation, but Integer Grid for State map
OBJ_EMPTY = 0
OBJ_WALL = 1
OBJ_APPLE = 2
OBJ_WASTE = 3 # Only for Cleanup
OBJ_AGENT = 4 # Represented dynamically, not in static map usually
OBJ_BEAM = 5  # Visual effect

@struct.dataclass
class AgentState:
    pos: chex.Array  # (2,) [y, x]
    dir: int         # 0~3
    reward: float
    frozen: int      # Frozen steps remaining (if zapped)
    last_action: int

@struct.dataclass
class GridState:
    grid: chex.Array # (H, W) or (H, W, C)
    agents: AgentState # Batched AgentState (struct of arrays)
    step_count: int

def move_forward(pos, direction):
    # direction: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    # y is row (0 is top), x is col (0 is left)
    # UP: y-1, RIGHT: x+1, DOWN: y+1, LEFT: x-1
    dy = jnp.array([-1, 0, 1, 0])
    dx = jnp.array([0, 1, 0, -1])
    
    new_y = pos[0] + dy[direction]
    new_x = pos[1] + dx[direction]
    return jnp.array([new_y, new_x])

def rotate(direction, action):
    # action 5: Left (-1), 6: Right (+1)
    # (dir + change) % 4
    change = jnp.where(action == ACTION_ROT_LEFT, -1, 1)
    return (direction + change) % 4

def check_collision(pos, grid, h, w):
    # Check Bounds
    in_bounds = (pos[0] >= 0) & (pos[0] < h) & (pos[1] >= 0) & (pos[1] < w)
    
    # Check Wall
    # Assuming grid contains object IDs
    # If out of bounds, treat as wall effectively (or just invalid)
    is_wall = jnp.where(in_bounds, grid[pos[0], pos[1]] == OBJ_WALL, True)
    
    return ~in_bounds | is_wall

def get_agent_view(grid, agent_pos, agent_dir, view_size=11):
    # Crop a window around agent, rotated to face up
    # This is expensive in plain numpy, JAX can optimize via lax.dynamic_slice or gather
    
    # 1. Pad Grid with Walls to handle out-of-bounds easily
    pad = view_size // 2
    padded_grid = jnp.pad(grid, pad, constant_values=OBJ_WALL)
    
    # 2. Slice (y, x) around agent
    # Agent pos is in original grid coords via (y, x)
    # In padded grid, center is (y+pad, x+pad)
    start_y = agent_pos[0]
    start_x = agent_pos[1]
    
    # dynamic_slice(operand, start_indices, slice_sizes)
    window = lax.dynamic_slice(padded_grid, (start_y, start_x), (view_size, view_size))
    
    # 3. Rotate so agent always faces UP
    # If dir=UP(0), rot=0
    # If dir=RIGHT(1), rot=1 (CCW 90? No, to make it Up, we rotate CW 90 -> k=3? or k=1?)
    # k is number of 90 degree rotations.
    # jnp.rot90 rotates counter-clockwise.
    # If facing Right(1), we want it Up. So rotate +90 deg (1 time CCW).
    # If facing Down(2), rotate 180 (2 times).
    # If facing Left(3), rotate 270 (3 times).
    
    # k is number of 90 degree rotations.
    # jnp.rot90 requires static k. agent_dir is Tracer in vmap.
    # Use lax.switch to handle dynamic k.
    
    k = agent_dir % 4
    
    def rot0(x): return x
    def rot1(x): return jnp.rot90(x, k=1)
    def rot2(x): return jnp.rot90(x, k=2)
    def rot3(x): return jnp.rot90(x, k=3)
    
    rotated_window = lax.switch(k, [rot0, rot1, rot2, rot3], window)
    
    return rotated_window

def process_movement(agent_state, action, grid, h, w):
    # Handles Forward/Backward/Left/Right/Rotate
    # Returns new_pos, new_dir
    
    # Rotate Logic
    is_rot = (action == ACTION_ROT_LEFT) | (action == ACTION_ROT_RIGHT)
    new_dir = jnp.where(is_rot, rotate(agent_state.dir, action), agent_state.dir)
    
    # Move Logic
    # 1: Forward, 2: Backward, 3: Left, 4: Right
    # Relative to current direction
    # Forward: dir
    # Backward: (dir + 2) % 4
    # Left: (dir + 3) % 4
    # Right: (dir + 1) % 4
    
    move_dir_map = jnp.array([
        0, # Noop
        0, # Fwd (+0)
        2, # Bwd (+2)
        3, # Left (+3)
        1, # Right (+1)
        0, 0, 0 # Rot/Beam
    ])
    
    # Current facing + Relative move
    move_dir = (agent_state.dir + move_dir_map[action]) % 4
    
    is_move = (action >= ACTION_FORWARD) & (action <= ACTION_RIGHT)
    potential_pos = move_forward(agent_state.pos, move_dir)
    
    # Check Collision with Map (Walls)
    # Note: Agent-Agent collision handled separately or assumed allowed/blocked?
    # Usually blocked. We need 'agent_map' passed in or handle sequentially.
    # For now, just Map Collision.
    collided = check_collision(potential_pos, grid, h, w)
    
    final_pos = jnp.where(is_move & ~collided, potential_pos, agent_state.pos)
    
    # Update State
    return agent_state.replace(pos=final_pos, dir=new_dir, last_action=action)

def diffuse_stencil(grid, diffusion_rate, decay_rate):
    """
    Simulate Pheromone Diffusion using 5-point stencil (Laplacian).
    grid: (H, W) or (H, W, C) - Pheromone concentrations
    diffusion_rate: Rate of spread
    decay_rate: Rate of evaporation
    """
    # Periodic Boundary Condition (Toroidal-like via roll)
    # 1. Neighbor Sum (Up, Down, Left, Right)
    # Axis 0 is Y (Height), Axis 1 is X (Width)
    # roll(shift=1, axis=0) -> moves row i to i+1 (Down visual, but index increase)
    # Let's align with move_forward logic if needed, but diffusion is isotropic usually.
    
    down = jnp.roll(grid, shift=1, axis=0)
    up   = jnp.roll(grid, shift=-1, axis=0)
    right = jnp.roll(grid, shift=1, axis=1)
    left  = jnp.roll(grid, shift=-1, axis=1)
    
    # Laplacian: neighbors - 4*center
    laplacian = (left + right + up + down) - 4 * grid
    
    # Diffusion Equation: dP/dt = D * Laplacian - Lambda * P
    # New P = P + rate * Laplacian - decay * P
    new_grid = grid + (diffusion_rate * laplacian) - (decay_rate * grid)
    
    return jnp.clip(new_grid, 0.0, 1.0)
