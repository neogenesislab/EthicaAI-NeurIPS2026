
import jax
import jax.numpy as jnp
from simulation.jax.agents.hrl_agent import (
    create_hrl_config, init_agent_state, calculate_drive, 
    update_internal_state, drive_reduction_reward,
    response_probability, update_thresholds
)
from simulation.jax.environments.common import diffuse_stencil

def test_hrl_logic():
    print("=== Testing HRL Logic ===")
    config = create_hrl_config(num_needs=2, setpoint_val=1.0)
    key = jax.random.PRNGKey(0)
    
    # 1. Init
    state = init_agent_state(config, key)
    print("Initial Levels:", state.levels)
    print("Initial Thresholds:", state.thresholds)
    initial_drive = calculate_drive(state, config)
    print("Initial Drive:", initial_drive)
    
    # 2. Decay (Worsen state)
    next_state = update_internal_state(state, jnp.zeros(2), config)
    print("Decayed Levels:", next_state.levels)
    
    # Reward should be negative
    reward = drive_reduction_reward(state, next_state, config)
    print("Reward (Decay):", reward)
    assert reward < 0.0, "Reward should be negative when state worsens"
    
    # 3. Recovery
    intake = config.setpoints - next_state.levels + config.decay_rate
    recovered_state = update_internal_state(next_state, intake, config)
    recov_reward = drive_reduction_reward(next_state, recovered_state, config)
    print("Reward (Recovery):", recov_reward)
    assert recov_reward > 0.0, "Reward should be positive when state improves"
    print("HRL Logic Passed!\n")

def test_threshold_model():
    print("=== Testing Response Threshold Model ===")
    config = create_hrl_config(num_tasks=2)
    
    # 1. Response Probability
    stimulus = jnp.array([0.5, 0.1]) # High waste, low apple
    threshold = jnp.array([0.3, 0.3]) # Equal thresholds
    
    probs = response_probability(stimulus, threshold)
    print("Stimulus:", stimulus)
    print("Threshold:", threshold)
    print("Response Probabilities:", probs)
    
    # High stimulus -> higher probability
    assert probs[0] > probs[1], "Higher stimulus should yield higher response probability"
    
    # 2. Low threshold -> higher probability
    threshold_low = jnp.array([0.1, 0.3])
    probs_low = response_probability(stimulus, threshold_low)
    print("Low-Threshold Probabilities:", probs_low)
    assert probs_low[0] > probs[0], "Lower threshold should yield higher probability for same stimulus"
    
    # 3. Threshold Update (Specialization)
    print("\n--- Threshold Specialization Test ---")
    threshold_start = jnp.array([0.5, 0.5])
    
    # Agent performs Task 0 (Cleaning) repeatedly
    performed = jnp.array([1.0, 0.0]) # Cleaned, did NOT harvest
    
    updated_threshold = threshold_start
    for i in range(50):
        updated_threshold = update_thresholds(updated_threshold, performed, config)
    
    print("After 50 steps (cleaning):", updated_threshold)
    assert updated_threshold[0] < threshold_start[0], "Cleaning threshold should decrease (specialization)"
    assert updated_threshold[1] > threshold_start[1], "Harvesting threshold should increase (de-specialization)"
    
    # 4. Division of Labor Emergence
    # Two agents starting with same thresholds, performing different tasks
    agent_a_thresh = jnp.array([0.5, 0.5])
    agent_b_thresh = jnp.array([0.5, 0.5])
    
    task_a = jnp.array([1.0, 0.0]) # Agent A cleans
    task_b = jnp.array([0.0, 1.0]) # Agent B harvests
    
    for _ in range(100):
        agent_a_thresh = update_thresholds(agent_a_thresh, task_a, config)
        agent_b_thresh = update_thresholds(agent_b_thresh, task_b, config)
    
    print("\nAgent A (Cleaner) Thresholds:", agent_a_thresh)
    print("Agent B (Harvester) Thresholds:", agent_b_thresh)
    
    # Agent A should be specialized in cleaning (low clean threshold)
    assert agent_a_thresh[0] < agent_b_thresh[0], "Agent A should have lower cleaning threshold"
    # Agent B should be specialized in harvesting (low harvest threshold)
    assert agent_b_thresh[1] < agent_a_thresh[1], "Agent B should have lower harvesting threshold"
    
    print("Threshold Model Passed!")

def test_diffusion_logic():
    print("\n=== Testing Diffusion Logic ===")
    H, W = 10, 10
    grid = jnp.zeros((H, W))
    grid = grid.at[5, 5].set(1.0)
    
    diffusion_rate = 0.1
    decay_rate = 0.01
    
    diffuse_jit = jax.jit(diffuse_stencil)
    new_grid = diffuse_jit(grid, diffusion_rate, decay_rate)
    
    print("Step 1 Center:", new_grid[5, 5])
    print("Step 1 Neighbor:", new_grid[5, 6])
    print("Step 1 Total Sum:", jnp.sum(new_grid))
    
    assert new_grid[5, 5] < 1.0, "Center should diffuse"
    assert new_grid[5, 6] > 0.0, "Neighbor should receive mass"
    assert jnp.sum(new_grid) <= jnp.sum(grid), "Total mass should not increase"
    
    print("Diffusion Logic Passed!")

if __name__ == "__main__":
    test_hrl_logic()
    test_threshold_model()
    test_diffusion_logic()
