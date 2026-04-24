import jax
import jax.numpy as jnp
from jax import random
import chex
from simulation.jax.training.train_pipeline import make_train
import time

def test_training_loop():
    print("Initializing Training Pipeline...")
    config = {
        "NUM_AGENTS": 5,
        "ENV_HEIGHT": 25,
        "ENV_WIDTH": 18,
        "MAX_STEPS": 100,
        "NUM_ENVS": 4,
        "NUM_UPDATES": 2,
        "ROLLOUT_LEN": 16,
        "LR": 2.5e-4,
        "HIDDEN_DIM": 128,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "MAX_GRAD_NORM": 0.5,
    }
    
    train_fn = make_train(config)
    
    print("JIT Compiling Train Function...")
    jit_train = jax.jit(train_fn)
    
    key = jax.random.PRNGKey(42)
    svo_theta = 0.7854 # 45 degrees (Prosocial)
    
    print("Running Training Loop...")
    start_time = time.time()
    result = jit_train(key, svo_theta)
    end_time = time.time()
    
    print("Training Loop Completed Successfully!")
    print(f"Time Taken: {end_time - start_time:.4f}s")
    
    # Unpack result: (runner_state, metrics_stack)
    runner_state, metrics = result
    train_state, obs, dones = runner_state
    
    print(f"\n=== Metrics (per epoch, {config['NUM_UPDATES']} epochs) ===")
    for key_name in ["reward_mean", "reward_std", "cooperation_rate", "gini",
                      "threshold_clean_mean", "threshold_harvest_mean",
                      "total_loss", "entropy"]:
        vals = metrics[key_name]
        print(f"  {key_name}: {vals}")
    
    print("\nAll metrics collected successfully!")

if __name__ == "__main__":
    test_training_loop()
