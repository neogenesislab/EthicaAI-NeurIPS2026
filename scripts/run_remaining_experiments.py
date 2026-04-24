#!/usr/bin/env python3
"""
EthicaAI — 나머지 2개 실험만 안전하게 실행.
GPU만 사용, CPU 분석 병렬 없음.
"""

import os
import sys
import json
import time
import copy
import logging
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
import numpy as np

from simulation.jax.training.train_pipeline import make_train

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("RemainingExps")

_platform = jax.default_backend()
log.info(f"JAX backend: {_platform}, Devices: {jax.devices()}")

OUTPUT_DIR = os.path.join(project_root, "experiments", "full_sweep_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 기본 설정 (Phase 1)
BASE_CONFIG = {
    "ENV_NAME": "cleanup",
    "NUM_AGENTS": 5,
    "ENV_HEIGHT": 15,
    "ENV_WIDTH": 15,
    "MAX_STEPS": 500,
    "NUM_ENVS": 128,  # 기존 01~05와 동일 조건 유지
    "NUM_UPDATES": 5000,
    "ROLLOUT_LEN": 500,
    "BATCH_SIZE": 256,
    "LR": 0.0003,
    "HIDDEN_DIM": 128,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENTROPY_COEFF": 0.05,
    "VF_COEFF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "HRL_NUM_NEEDS": 2,
    "HRL_NUM_TASKS": 2,
    "HRL_ALPHA": 0.0,
    "HRL_THRESH_INCREASE": 0.005,
    "HRL_THRESH_DECREASE": 0.05,
    "HRL_INTAKE_VAL": 0.2,
    "REWARD_APPLE": 1.0,
    "COST_BEAM": -0.1,
    "META_BETA": 0.1,
    "META_SURVIVAL_THRESHOLD": -5.0,
    "META_WEALTH_BOOST": 5.0,
    "META_LAMBDA_EMA": 0.9,
    "USE_META_RANKING": False,
    "META_USE_DYNAMIC_LAMBDA": False,
    "GENESIS_MODE": False,
    "USE_INEQUITY_AVERSION": False,
    "GENESIS_BETA_BASE": 10.0,
    "GENESIS_GAMMA": 2.0,
    "GENESIS_ALPHA": 0.3,
    "GENESIS_BETA": 0.7,
    "GENESIS_LOGIC_MODE": "adaptive_beta",
    "LOG_INTERVAL": 50,
}

# 남은 2개 실험만
REMAINING = {
    "06_inequity_aversion": {
        "desc": "MAPPO + Inequity Aversion (SA-PPO)",
        "overrides": {
            "USE_INEQUITY_AVERSION": True,
            "IA_ALPHA": 1.0,
            "IA_BETA": 0.1,
            "IA_EMA_LAMBDA": 0.95,
        }
    },
    "07_genesis_full": {
        "desc": "MAPPO + Genesis 전체 (Adaptive Beta)",
        "overrides": {
            "HRL_ALPHA": 1.0,
            "USE_META_RANKING": True,
            "META_USE_DYNAMIC_LAMBDA": True,
            "GENESIS_MODE": True,
        }
    },
}

SEEDS = [42, 123, 7]
SVO_CONDITIONS = {
    "Prosocial": float(jnp.pi / 4),
    "Individualist": 0.0,
}


def run_experiment(exp_name, config, seeds, svo_conditions):
    """하나의 실험 실행."""
    log.info(f"{'='*60}")
    log.info(f"실험: {exp_name}")
    log.info(f"설명: {config.get('_desc', '')}")
    log.info(f"{'='*60}")
    
    log.info("JAX 그래프 컴파일 중...")
    compile_start = time.time()
    train_fn = make_train(config)
    train_fn = jax.jit(train_fn)
    compile_time = time.time() - compile_start
    log.info(f"컴파일 완료: {compile_time:.1f}초")
    
    results = {
        "experiment": exp_name,
        "config": {k: v for k, v in config.items() if k != '_desc'},
        "compile_time_sec": compile_time,
        "conditions": {}
    }
    
    for svo_name, svo_val in svo_conditions.items():
        log.info(f"  조건: {svo_name} (θ={svo_val:.4f})")
        seed_results = []
        
        for seed in seeds:
            log.info(f"    시드: {seed}...")
            run_start = time.time()
            
            try:
                key = jax.random.PRNGKey(seed)
                runner_state, metrics_history = train_fn(key, float(svo_val))
                
                coop_rate = float(metrics_history["cooperation_rate"][-10:].mean())
                reward_mean = float(metrics_history["reward_mean"][-10:].mean())
                gini = float(metrics_history["gini"][-10:].mean())
                
                total_updates = len(metrics_history["reward_mean"])
                sample_indices = list(range(0, total_updates, max(1, total_updates // 50)))
                
                run_time = time.time() - run_start
                
                seed_result = {
                    "seed": seed,
                    "cooperation_rate": coop_rate,
                    "reward_mean": reward_mean,
                    "gini": gini,
                    "run_time_sec": run_time,
                    "learning_curve": {
                        "reward_mean": [float(metrics_history["reward_mean"][i]) for i in sample_indices],
                        "cooperation_rate": [float(metrics_history["cooperation_rate"][i]) for i in sample_indices],
                        "gini": [float(metrics_history["gini"][i]) for i in sample_indices],
                        "update_indices": sample_indices,
                    }
                }
                seed_results.append(seed_result)
                log.info(f"    ✅ Seed {seed}: Coop={coop_rate:.4f}, Reward={reward_mean:.4f}, "
                        f"Gini={gini:.4f} ({run_time:.0f}초)")
            except Exception as e:
                run_time = time.time() - run_start
                log.error(f"    ❌ Seed {seed}: {str(e)[:100]} ({run_time:.0f}초)")
                seed_results.append({"seed": seed, "error": str(e), "run_time_sec": run_time})
        
        valid = [r for r in seed_results if "error" not in r]
        if valid:
            condition_summary = {
                "seeds": seed_results,
                "mean_cooperation": float(np.mean([r["cooperation_rate"] for r in valid])),
                "std_cooperation": float(np.std([r["cooperation_rate"] for r in valid])),
                "mean_reward": float(np.mean([r["reward_mean"] for r in valid])),
                "std_reward": float(np.std([r["reward_mean"] for r in valid])),
                "mean_gini": float(np.mean([r["gini"] for r in valid])),
                "n_successful": len(valid),
                "n_failed": len(seed_results) - len(valid),
            }
        else:
            condition_summary = {"seeds": seed_results, "n_successful": 0, "n_failed": len(seed_results), "error": "모든 시드 실패"}
        
        results["conditions"][svo_name] = condition_summary
    
    return results


def main():
    start = datetime.now()
    log.info(f"나머지 2개 실험 시작: {start.strftime('%H:%M:%S')}")
    
    for exp_name, exp_def in REMAINING.items():
        # 이미 완료된 결과 스킵
        out_file = os.path.join(OUTPUT_DIR, f"{exp_name}.json")
        if os.path.exists(out_file):
            log.info(f"⏩ {exp_name} 이미 완료 — 스킵")
            continue
        
        config = copy.deepcopy(BASE_CONFIG)
        config.update(exp_def["overrides"])
        config["_desc"] = exp_def["desc"]
        
        result = run_experiment(exp_name, config, SEEDS, SVO_CONDITIONS)
        
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.info(f"💾 저장: {out_file}")
    
    end = datetime.now()
    log.info(f"\n🏁 완료! 소요: {end - start}")


if __name__ == "__main__":
    main()
