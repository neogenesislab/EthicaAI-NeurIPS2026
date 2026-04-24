"""
Genesis Run #004: Medium Scale Comparative Experiment
======================================================
Runs Original vs Adaptive Beta vs Inverse Beta at Medium Scale
(NUM_UPDATES=300, 20 agents) for a proper hypothesis test.
"""
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from simulation.jax.experiment_jax import run_sweep
from simulation.jax.config import SVO_SWEEP_THETAS
import jax.numpy as jnp


def extract_summary(all_results):
    summaries = {}
    for svo_name, data in all_results.items():
        runs = data["runs"]
        avg_coop = float(jnp.mean(jnp.array([r["metrics"]["cooperation_rate"][-1] for r in runs])))
        avg_reward = float(jnp.mean(jnp.array([r["metrics"]["reward_mean"][-1] for r in runs])))
        avg_gini = float(jnp.mean(jnp.array([r["metrics"]["gini"][-1] for r in runs])))
        summaries[svo_name] = {"coop": avg_coop, "reward": avg_reward, "gini": avg_gini}
    return summaries


def run_genesis_004():
    print("=" * 60)
    print("  Genesis #004: Medium Scale (300 updates, 20 agents)")
    print("  Comparing: Original | Adaptive Beta | Inverse Beta")
    print("=" * 60)

    output_dir = "simulation/outputs/genesis_004"
    os.makedirs(output_dir, exist_ok=True)

    # Full Altruist + Prosocial for comparison
    angles = {
        "prosocial": SVO_SWEEP_THETAS["prosocial"],
        "full_altruist": SVO_SWEEP_THETAS["full_altruist"],
    }
    seeds = [42, 1004]
    all_summaries = {}

    modes = [
        ("original", {}),
        ("adaptive_beta", {"GENESIS_MODE": True, "GENESIS_HYPOTHESIS": "adaptive_beta"}),
        ("inverse_beta", {"GENESIS_MODE": True, "GENESIS_HYPOTHESIS": "inverse_beta"}),
    ]

    for i, (mode_name, override) in enumerate(modes, 1):
        print(f"\n[{i}/3] {mode_name}...")
        res, _ = run_sweep(
            scale="medium", svo_angles=angles, seeds=seeds,
            output_dir=os.path.join(output_dir, mode_name),
            config_override=override if override else None
        )
        s = extract_summary(res)
        all_summaries[mode_name] = s
        for svo, metrics in s.items():
            print(f"  {svo:15s} | Coop: {metrics['coop']:.4f} | Reward: {metrics['reward']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("  GENESIS #004 MEDIUM SCALE RESULTS")
    print("=" * 70)
    print(f"  {'Mode':20s} | {'SVO':15s} | {'Coop':>8s} | {'Reward':>8s} | {'Gini':>8s}")
    print("-" * 70)
    for mode, svo_data in all_summaries.items():
        for svo, m in svo_data.items():
            print(f"  {mode:20s} | {svo:15s} | {m['coop']:8.4f} | {m['reward']:8.4f} | {m['gini']:8.4f}")
    print("=" * 70)

    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved: {output_dir}/comparison.json")


if __name__ == "__main__":
    run_genesis_004()
