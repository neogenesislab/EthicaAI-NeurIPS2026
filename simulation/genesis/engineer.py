import os
import json
import logging
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from simulation.jax.config import get_config
from simulation.jax.training.train_pipeline import make_train

# v2.0: GPU 자동 감지 (CPU 강제 제거)
# GPU가 있으면 자동 사용, 없으면 CPU 폴백
_platform = jax.default_backend()
logging.basicConfig(level=logging.INFO)
logging.info(f"JAX backend: {_platform}, Devices: {jax.devices()}")
print(f"🖥️ JAX Platform: {_platform} | Devices: {jax.devices()}")

class Engineer:
    def __init__(self, config_path="experiments/evolution/current/config.json"):
        self.config_path = config_path
        self.result_path = "experiments/evolution/current/result.json"
        self.is_gpu = jax.default_backend() == "gpu"

    def load_config(self):
        with open(self.config_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    def run_simulation(self):
        config = self.load_config()
        print(f"🛠️ Engineer: Running simulation with config: {json.dumps(config, indent=2)}")
        
        # v2.1: config.json 값 우선, 기본값만 폴백 (하드코딩 금지 원칙)
        if self.is_gpu:
            config.setdefault("NUM_ENVS", 128)
            config.setdefault("BATCH_SIZE", 256)
            config.setdefault("NUM_UPDATES", 5000)
            config.setdefault("LOG_INTERVAL", 50)
            print(f"  > 🚀 GPU Mode: NUM_ENVS={config['NUM_ENVS']}, "
                  f"BATCH={config['BATCH_SIZE']}, UPDATES={config['NUM_UPDATES']}")
        else:
            config.setdefault("NUM_ENVS", 16)
            config.setdefault("BATCH_SIZE", 256)
            config.setdefault("NUM_UPDATES", 1000)
            config.setdefault("LOG_INTERVAL", 10)
            print(f"  > 🐢 CPU Mode: NUM_ENVS={config['NUM_ENVS']}, "
                  f"BATCH={config['BATCH_SIZE']}, UPDATES={config['NUM_UPDATES']}")

        config.setdefault("SEEDS", [42, 123, 7])
        # v2.1: GENESIS_MODE는 config.json 값 존중 (더 이상 강제 True 아님)
        config.setdefault("GENESIS_MODE", False)

        # Prepare Result Container
        results = {}
        
        # Compile Train Function
        print("  > Compiling JAX graph...")
        train_fn = make_train(config)
        train_fn = jax.jit(train_fn)
        
        # Test 2 Conditions: Prosocial & Individualist
        svo_conditions = {
            "Prosocial": jnp.pi/4,      # 45 degrees
            "Individualist": 0.0        # 0 degrees
        }
        
        print(f"  > Starting loops (JIT ENABLED, {_platform.upper()})...", flush=True)
        
        for svo_name, svo_val in svo_conditions.items():
            print(f"  > Testing {svo_name}...", end="", flush=True)
            try:
                # v2.0: 다중 시드 실행 (헌법 제4조 — 최소 3회 반복)
                seed_coop_rates = []
                seed_rewards = []
                seed_ginis = []

                for seed in config["SEEDS"]:
                    print(f" [Seed={seed}]", end="", flush=True)
                    key = jax.random.PRNGKey(seed)
                    runner_state, metrics_history = train_fn(key, float(svo_val))
                    
                    coop = float(metrics_history["cooperation_rate"][-10:].mean())
                    rew = float(metrics_history["reward_mean"][-10:].mean())
                    gini = float(metrics_history["gini"][-10:].mean())
                    
                    seed_coop_rates.append(coop)
                    seed_rewards.append(rew)
                    seed_ginis.append(gini)

                # v2.0: 다중 시드 평균 + 표준편차 (헌법 제5조 — 전체 분포 보고)
                coop_mean = float(np.mean(seed_coop_rates))
                coop_std = float(np.std(seed_coop_rates))
                reward_mean = float(np.mean(seed_rewards))
                gini_mean = float(np.mean(seed_ginis))
                
                results[svo_name] = {
                    "cooperation_rate": coop_mean,
                    "cooperation_std": coop_std,
                    "cooperation_per_seed": seed_coop_rates,
                    "reward_mean": reward_mean,
                    "gini": gini_mean,
                    "num_seeds": len(config["SEEDS"]),
                    "platform": _platform,
                }
                print(f" Done. Coop={coop_mean:.4f}±{coop_std:.4f}")
                
            except Exception as e:
                print(f" Failed: {e}")
                results[svo_name] = {"error": str(e)}

        # Save Result
        with open(self.result_path, "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"💾 Results saved to {self.result_path}")
        return results

if __name__ == "__main__":
    engineer = Engineer()
    engineer.run_simulation()
