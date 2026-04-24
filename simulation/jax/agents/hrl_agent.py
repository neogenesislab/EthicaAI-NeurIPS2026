
import jax
import jax.numpy as jnp
from flax import struct
import chex
from typing import Tuple

@struct.dataclass
class HRLConfig:
    """Configuration for Homeostatic Agent"""
    # Physiology
    num_needs: int = 2 # e.g., Energy, Satiety
    setpoints: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([1.0, 1.0]))
    
    # Drive Parameters
    m: float = 2.0 # Minkowski metric order (Euclidean=2)
    n: float = 4.0 # Non-linearity (Higher n = rapid increase near death)
    
    # Dynamics
    decay_rate: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([0.01, 0.005]))
    
    # Threshold Parameters (Response Threshold Model)
    num_tasks: int = 2 # e.g., Clean, Harvest
    threshold_increase: float = 0.005 # xi+ (de-specialization when NOT performing task)
    threshold_decrease: float = 0.05  # xi- (specialization when performing task)
    
    # Intake & Threshold Initialization
    intake_val: float = 0.2           # Resource intake amount
    threshold_init_min: float = 0.3   # Initial threshold lower bound
    threshold_init_max: float = 0.7   # Initial threshold upper bound
    
@struct.dataclass
class InternalState:
    """Physiological State of the Agent"""
    levels: jnp.ndarray # (num_needs,) - Current levels (0.0 ~ 1.0)
    thresholds: jnp.ndarray # (num_tasks,) - Response thresholds per task (0.0 ~ 1.0)
    
def create_hrl_config(num_needs=2, setpoint_val=1.0, decay_val=0.01, num_tasks=2,
                      threshold_increase=0.005, threshold_decrease=0.05,
                      intake_val=0.2, threshold_init_min=0.3, threshold_init_max=0.7):
    return HRLConfig(
        num_needs=num_needs,
        setpoints=jnp.full((num_needs,), setpoint_val),
        decay_rate=jnp.full((num_needs,), decay_val),
        num_tasks=num_tasks,
        threshold_increase=threshold_increase,
        threshold_decrease=threshold_decrease,
        intake_val=intake_val,
        threshold_init_min=threshold_init_min,
        threshold_init_max=threshold_init_max,
    )

def init_agent_state(config: HRLConfig, rng: chex.PRNGKey) -> InternalState:
    """Initialize agent internal state near setpoints with small noise"""
    k1, k2 = jax.random.split(rng)
    noise = jax.random.uniform(k1, shape=(config.num_needs,), minval=-0.1, maxval=0.0)
    # Initialize thresholds randomly for diversity (Division of Labor)
    init_thresholds = jax.random.uniform(
        k2, shape=(config.num_tasks,), 
        minval=config.threshold_init_min, 
        maxval=config.threshold_init_max
    )
    return InternalState(
        levels=jnp.clip(config.setpoints + noise, 0.0, 1.0),
        thresholds=init_thresholds
    )

def calculate_drive(state: InternalState, config: HRLConfig) -> float:
    """
    Calculate Homeostatic Drive (Distance from Setpoint).
    D(H_t) = sum( |h* - h_t|^n )^(1/m)
    """
    deviation = jnp.abs(config.setpoints - state.levels)
    drive = jnp.sum(jnp.power(deviation, config.n))
    drive = jnp.power(drive, 1.0 / config.m)
    return drive

def response_probability(stimulus: jnp.ndarray, threshold: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate Response Probability for each task (Response Threshold Model).
    P(task_j) = s_j^2 / (s_j^2 + theta_j^2)
    
    Where:
    - s_j: Stimulus intensity for task j (e.g., waste density for cleaning)
    - theta_j: Agent's personal threshold for task j
    
    High stimulus & low threshold -> high probability (specialist)
    Low stimulus & high threshold -> low probability (non-specialist)
    
    Returns: (num_tasks,) probabilities
    """
    s2 = jnp.square(stimulus)
    t2 = jnp.square(threshold)
    return s2 / (s2 + t2 + 1e-8)

def update_thresholds(
    thresholds: jnp.ndarray, 
    performed_tasks: jnp.ndarray,
    config: HRLConfig
) -> jnp.ndarray:
    """
    Update Response Thresholds based on task performance.
    
    - If agent performed task j: theta_j decreases (specialization)
    - If agent did NOT perform task j: theta_j increases (de-specialization)
    
    theta_j = theta_j - xi- * performed + xi+ * (1 - performed)
    
    performed_tasks: (num_tasks,) boolean/float array (1.0 if performed, 0.0 if not)
    """
    decrease = config.threshold_decrease * performed_tasks
    increase = config.threshold_increase * (1.0 - performed_tasks)
    new_thresholds = thresholds - decrease + increase
    return jnp.clip(new_thresholds, 0.01, 1.0) # Min 0.01 to prevent division issues

def update_internal_state(
    state: InternalState, 
    intake: jnp.ndarray, 
    config: HRLConfig
) -> InternalState:
    """
    Update internal state based on intake and natural decay.
    h_{t+1} = h_t - decay + intake
    """
    new_levels = state.levels - config.decay_rate + intake
    new_levels = jnp.clip(new_levels, 0.0, 1.0)
    return state.replace(levels=new_levels)

def drive_reduction_reward(
    prev_state: InternalState, 
    next_state: InternalState, 
    config: HRLConfig
) -> float:
    """
    Calculate Reward based on Drive Reduction Theory.
    r = D(H_t) - D(H_{t+1})
    """
    prev_drive = calculate_drive(prev_state, config)
    next_drive = calculate_drive(next_state, config)
    return prev_drive - next_drive
