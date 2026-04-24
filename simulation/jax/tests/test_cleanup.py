import jax
import jax.numpy as jnp
from jax import random
import chex
import traceback
from simulation.jax.environments.cleanup import CleanupJax, EnvParams, ACTION_NOOP, ACTION_BEAM

def test_cleanup_env():
    try:
        # 1. Setup
        key = random.PRNGKey(42)
        env = CleanupJax()
        params = EnvParams()
        
        # 2. Reset
        print("Running Reset...")
        obs, state = env.reset(key, params)
        print("Reset Successful")
        print("Grid Shape:", state.grid.shape)
        
        # 3. Step (No-op)
        print("Running Step (No-op)...")
        actions_jax = {a: jnp.array(ACTION_NOOP) for a in env.agents}
        
        key, step_key = random.split(key)
        obs, state, rewards, dones, info = env.step(step_key, state, actions_jax, params)
        print("Step 1 (No-op) Successful")
        print("Rewards:", rewards)
        
        # 4. Step (Beam)
        print("Running Step (Beam)...")
        actions_beam = {a: jnp.array(ACTION_BEAM) for a in env.agents}
        key, step_key = random.split(key)
        obs, state, rewards, dones, info = env.step(step_key, state, actions_beam, params)
        print("Step 2 (Beam) Successful")
        print("Cleaned Count:", info["cleaned_count"])
        
        # 5. JIT Compile Test
        print("Testing JIT Compilation...")
        jit_step = jax.jit(env.step)
        
        # Need to ensure params is PyTree or static?
        # EnvParams is struct.dataclass -> PyTree -> Tracer.
        # But fields are float/int. 
        # Inside step, we don't use params for shape anymore. So it should be fine as Tracer.
        
        jit_step(step_key, state, actions_beam, params)
        print("JIT Compilation Successful")
        
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test_cleanup_env()
