"""
Genesis Run #003: Properly Connected Beta → Lambda Experiment
==============================================================
Compares Original, Adaptive Beta, and Inverse Beta hypotheses.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from simulation.jax.experiment_jax import run_sweep
from simulation.jax.config import SVO_SWEEP_THETAS, CONFIG_SMALL
import jax.numpy as jnp


def extract_summary(all_results):
    """run_sweep 반환값에서 핵심 지표 추출."""
    summaries = {}
    for svo_name, data in all_results.items():
        runs = data["runs"]
        avg_coop = float(jnp.mean(jnp.array([r["metrics"]["cooperation_rate"][-1] for r in runs])))
        avg_reward = float(jnp.mean(jnp.array([r["metrics"]["reward_mean"][-1] for r in runs])))
        avg_gini = float(jnp.mean(jnp.array([r["metrics"]["gini"][-1] for r in runs])))
        summaries[svo_name] = {"coop": avg_coop, "reward": avg_reward, "gini": avg_gini}
    return summaries


def run_genesis_003():
    print("=" * 60)
    print("  Genesis Run #003: Fixed Beta->Lambda Connection")
    print("  Comparing: Original | Adaptive Beta | Inverse Beta")
    print("=" * 60)

    output_dir = "simulation/outputs/genesis_003"
    os.makedirs(output_dir, exist_ok=True)
    angles = {"full_altruist": SVO_SWEEP_THETAS["full_altruist"]}
    seeds = [1004]
    all_summaries = {}

    # --- Run 1: Original (no Genesis) ---
    print("\n[1/3] Original Meta-Ranking...")
    res, _ = run_sweep(scale="small", svo_angles=angles, seeds=seeds,
                       output_dir=os.path.join(output_dir, "original"))
    s = extract_summary(res)["full_altruist"]
    all_summaries["original"] = s
    print(f"  => Coop: {s['coop']:.4f} | Reward: {s['reward']:.4f} | Gini: {s['gini']:.4f}")

    # --- Run 2: Adaptive Beta (dampening) ---
    print("\n[2/3] Genesis Adaptive Beta (Dampening)...")
    override = {"GENESIS_MODE": True, "GENESIS_HYPOTHESIS": "adaptive_beta"}
    res, _ = run_sweep(scale="small", svo_angles=angles, seeds=seeds,
                       output_dir=os.path.join(output_dir, "adaptive"),
                       config_override=override)
    s = extract_summary(res)["full_altruist"]
    all_summaries["adaptive_beta"] = s
    print(f"  => Coop: {s['coop']:.4f} | Reward: {s['reward']:.4f} | Gini: {s['gini']:.4f}")

    # --- Run 3: Inverse Beta (crisis response) ---
    print("\n[3/3] Genesis Inverse Beta (Crisis Response)...")
    override = {"GENESIS_MODE": True, "GENESIS_HYPOTHESIS": "inverse_beta"}
    res, _ = run_sweep(scale="small", svo_angles=angles, seeds=seeds,
                       output_dir=os.path.join(output_dir, "inverse"),
                       config_override=override)
    s = extract_summary(res)["full_altruist"]
    all_summaries["inverse_beta"] = s
    print(f"  => Coop: {s['coop']:.4f} | Reward: {s['reward']:.4f} | Gini: {s['gini']:.4f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  GENESIS #003 COMPARATIVE RESULTS")
    print("=" * 60)
    print(f"  {'Mode':20s} | {'Coop':>8s} | {'Reward':>8s} | {'Gini':>8s}")
    print("-" * 60)
    for mode, s in all_summaries.items():
        print(f"  {mode:20s} | {s['coop']:8.4f} | {s['reward']:8.4f} | {s['gini']:8.4f}")
    print("=" * 60)

    # Save
    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved: {output_dir}/comparison.json")


if __name__ == "__main__":
    run_genesis_003()
