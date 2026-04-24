"""
JAX Training Pipeline - Scale Benchmark
Tests performance at different agent counts / environment sizes.
"""
import jax
import jax.numpy as jnp
import time
from simulation.jax.training.train_pipeline import make_train
from simulation.jax.config import get_config, SVO_SWEEP_THETAS

def run_benchmark(scale="small"):
    print(f"\n{'='*50}")
    print(f"BENCHMARK: {scale.upper()} SCALE")
    print(f"{'='*50}")
    
    config = get_config(scale)
    print(f"Agents: {config['NUM_AGENTS']}")
    print(f"Map: {config['ENV_HEIGHT']}x{config['ENV_WIDTH']}")
    print(f"Envs: {config['NUM_ENVS']}")
    print(f"Updates: {config['NUM_UPDATES']}")
    print(f"Rollout: {config['ROLLOUT_LEN']}")
    
    train_fn = make_train(config)
    jit_train = jax.jit(train_fn)
    
    key = jax.random.PRNGKey(42)
    svo_theta = SVO_SWEEP_THETAS["fair"] # 45 degrees
    
    # Warmup (JIT compile)
    print("\nJIT Compiling...")
    t0 = time.time()
    result = jit_train(key, svo_theta)
    # Force synchronous execution for accurate timing
    jax.block_until_ready(result)
    compile_time = time.time() - t0
    print(f"JIT Compile Time: {compile_time:.2f}s")
    
    # Benchmark (Already compiled, measure pure execution)
    print("Running Benchmark (3 runs)...")
    times = []
    for i in range(3):
        key = jax.random.PRNGKey(i + 100)
        t0 = time.time()
        result = jit_train(key, svo_theta)
        jax.block_until_ready(result)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage Execution Time: {avg_time:.4f}s")
    print(f"Steps/Second: {config['NUM_UPDATES'] * config['ROLLOUT_LEN'] * config['NUM_ENVS'] / avg_time:.0f}")
    
    return avg_time

if __name__ == "__main__":
    # Run Small (Quick validation)
    run_benchmark("small")
    
    # Run Medium (Main experiment scale)
    run_benchmark("medium")
    
    # Large is optional (may require more memory)
    # run_benchmark("large")
    
    print("\n\nBenchmark Complete!")
