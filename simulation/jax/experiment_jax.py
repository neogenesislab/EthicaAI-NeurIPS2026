"""
EthicaAI JAX Experiment Runner
SVO Sweep × Multiple Seeds → MetricsBundle per condition.
"""
import jax
import jax.numpy as jnp
from jax import random
import time
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List

from simulation.jax.training.train_pipeline import make_train
from simulation.jax.config import get_config, SVO_SWEEP_THETAS

# 하드코딩 금지: config.py의 SVO_SWEEP_THETAS를 단일 진실의 원천으로 사용
SVO_ANGLES = SVO_SWEEP_THETAS

def run_single(config, svo_theta, seed):
    """Run a single training experiment and return metrics."""
    train_fn = make_train(config)
    jit_train = jax.jit(train_fn)
    
    key = random.PRNGKey(seed)
    
    t0 = time.time()
    result = jit_train(key, svo_theta)
    jax.block_until_ready(result)
    elapsed = time.time() - t0
    
    runner_state, metrics_stack = result
    train_state, obs, dones = runner_state
    
    # Extract final HRL state
    final_thresholds = train_state.hrl_state.thresholds # (B, N, Tasks)
    
    return {
        "metrics": jax.tree_util.tree_map(lambda x: x.tolist(), metrics_stack),
        "final_thresholds_mean": {
            "clean": float(final_thresholds[:, :, 0].mean()),
            "harvest": float(final_thresholds[:, :, 1].mean()),
        },
        "elapsed_time": elapsed,
    }

def run_sweep(scale="small", svo_angles=None, seeds=None, output_dir=None, config_override=None):
    """
    Run SVO Sweep experiment.
    
    Args:
        scale: "small", "medium", or "large"
        svo_angles: Dict of {name: theta_radians}
        seeds: List of random seeds
        output_dir: Directory to save results
        config_override: Dict to override config values (e.g. {"USE_META_RANKING": False})
    """
    if svo_angles is None:
        svo_angles = SVO_ANGLES
    if seeds is None:
        seeds = [0, 42, 123]
    if output_dir is None:
        output_dir = "simulation/outputs/sweep"
    
    config = get_config(scale)
    
    # Config Override 적용 (Baseline/Ablation 지원)
    if config_override:
        config.update(config_override)
    
    print("=" * 60)
    print(f"EthicaAI SVO Sweep Experiment")
    print(f"Scale: {scale} | Agents: {config['NUM_AGENTS']}")
    print(f"SVO Conditions: {len(svo_angles)} | Seeds: {len(seeds)}")
    print(f"Total Runs: {len(svo_angles) * len(seeds)}")
    print("=" * 60)
    
    all_results = {}
    
    for svo_name, svo_theta in svo_angles.items():
        print(f"\n--- SVO: {svo_name} (θ={svo_theta:.4f} rad) ---")
        seed_results = []
        
        for seed in seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            result = run_single(config, svo_theta, seed)
            print(f"Done ({result['elapsed_time']:.2f}s)")
            seed_results.append(result)
        
        all_results[svo_name] = {
            "theta": svo_theta,
            "seeds": seeds,
            "runs": seed_results,
        }
        
        # Print Summary for this SVO
        avg_reward = jnp.mean(jnp.array([
            r["metrics"]["reward_mean"][-1] for r in seed_results
        ]))
        avg_coop = jnp.mean(jnp.array([
            r["metrics"]["cooperation_rate"][-1] for r in seed_results
        ]))
        avg_gini = jnp.mean(jnp.array([
            r["metrics"]["gini"][-1] for r in seed_results
        ]))
        print(f"  → Reward: {avg_reward:.4f} | Coop: {avg_coop:.4f} | Gini: {avg_gini:.4f}")
    
    # Save Results
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"sweep_{scale}_{int(time.time())}.json")
    
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {result_path}")
    print(f"{'='*60}")
    
    return all_results, result_path


def print_summary_table(results):
    """Print a summary comparison table."""
    print("\n" + "=" * 70)
    print(f"{'SVO Condition':<16} {'θ(rad)':<8} {'Reward':<10} {'Coop%':<8} {'Gini':<8} {'θ_clean':<8} {'θ_harv':<8}")
    print("-" * 70)
    
    for name, data in results.items():
        runs = data["runs"]
        avg_r = jnp.mean(jnp.array([r["metrics"]["reward_mean"][-1] for r in runs]))
        avg_c = jnp.mean(jnp.array([r["metrics"]["cooperation_rate"][-1] for r in runs]))
        avg_g = jnp.mean(jnp.array([r["metrics"]["gini"][-1] for r in runs]))
        avg_tc = jnp.mean(jnp.array([r["final_thresholds_mean"]["clean"] for r in runs]))
        avg_th = jnp.mean(jnp.array([r["final_thresholds_mean"]["harvest"] for r in runs]))
        
        print(f"{name:<16} {data['theta']:<8.4f} {avg_r:<10.4f} {avg_c:<8.4f} {avg_g:<8.4f} {avg_tc:<8.4f} {avg_th:<8.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Quick Test: Small scale, 3 SVO angles, 2 seeds
    quick_angles = {
        "selfish": SVO_ANGLES["selfish"],
        "prosocial": SVO_ANGLES["prosocial"],
        "altruistic": SVO_ANGLES["full_altruist"],
    }
    
    results, path = run_sweep(
        scale="small",
        svo_angles=quick_angles,
        seeds=[0, 42],
    )
    
    print_summary_table(results)
