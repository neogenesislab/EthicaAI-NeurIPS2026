#!/usr/bin/env python3
"""
EthicaAI Genesis — Phase 1 Full Sweep 종합 학습 스크립트.

목적: 한 번 실행으로 모든 실험 조건의 데이터를 수집하여 재실행 불필요하도록 함.
예상 소요: RTX 4070 SUPER 12GB 기준 약 3~6시간.

실행 방법:
    source ~/ethicaai_env/bin/activate
    cd /mnt/d/00.test/PAPER/EthicaAI_anon2
    python3 scripts/full_sweep_phase1.py 2>&1 | tee experiments/full_sweep_log.txt

실험 목록 (총 7개 조건 × 3 시드 × 2 SVO = 42회 학습):
    1. Pure MAPPO (Baseline) — 모든 모듈 OFF
    2. MAPPO + Reward Shaping — 청소 보너스 +0.01
    3. MAPPO + HRL (α=1.0) — 내적 동기 ON
    4. MAPPO + Meta-Ranking — 불평등 기반 보상
    5. MAPPO + Meta-Ranking + Dynamic λ — 자원 의존적 이타성
    6. MAPPO + IA (Inequity Aversion) — 불공정 회피
    7. MAPPO + Genesis (Adaptive Beta) — Genesis 전체 ON
"""

import os
import sys
import json
import time
import copy
import logging
import traceback
from datetime import datetime, timedelta

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
import numpy as np

from simulation.jax.training.train_pipeline import make_train

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("FullSweep")

# GPU 확인
_platform = jax.default_backend()
log.info(f"JAX backend: {_platform}, Devices: {jax.devices()}")

# ========== 기본 설정 (Phase 1 Baseline) ==========
BASE_CONFIG = {
    "ENV_NAME": "cleanup",
    "NUM_AGENTS": 5,
    "ENV_HEIGHT": 15,
    "ENV_WIDTH": 15,
    "MAX_STEPS": 500,
    "NUM_ENVS": 128,
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

# ========== 실험 조건 정의 ==========
EXPERIMENTS = {
    "01_pure_mappo": {
        "desc": "순수 MAPPO Baseline — 모든 모듈 OFF",
        "overrides": {}
    },
    "02_reward_shaping": {
        "desc": "MAPPO + 청소 보상 (COST_BEAM=-0.1 → +0.01)",
        "overrides": {
            "COST_BEAM": 0.01,  # 청소에 미세 보상
        }
    },
    "03_hrl_only": {
        "desc": "MAPPO + HRL 내적 동기 (α=1.0)",
        "overrides": {
            "HRL_ALPHA": 1.0,
        }
    },
    "04_meta_ranking": {
        "desc": "MAPPO + Meta-Ranking (정적 λ)",
        "overrides": {
            "USE_META_RANKING": True,
            "META_USE_DYNAMIC_LAMBDA": False,
        }
    },
    "05_meta_dynamic_lambda": {
        "desc": "MAPPO + Meta-Ranking + Dynamic λ",
        "overrides": {
            "USE_META_RANKING": True,
            "META_USE_DYNAMIC_LAMBDA": True,
        }
    },
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
    "Prosocial": float(jnp.pi / 4),    # 45도
    "Individualist": 0.0,               # 0도
}

# ========== 결과 저장 경로 ==========
OUTPUT_DIR = os.path.join(project_root, "experiments", "full_sweep_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_single_experiment(exp_name, config, seeds, svo_conditions):
    """하나의 실험 조건에 대해 모든 시드와 SVO 조건을 실행합니다."""
    log.info(f"{'='*60}")
    log.info(f"실험: {exp_name}")
    log.info(f"설명: {config.get('_desc', '')}")
    log.info(f"{'='*60}")
    
    # JIT 컴파일 (실험 조건당 1회)
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
            log.info(f"    시드: {seed}...", )
            run_start = time.time()
            
            try:
                key = jax.random.PRNGKey(seed)
                runner_state, metrics_history = train_fn(key, float(svo_val))
                
                # 지표 추출
                coop_rate = float(metrics_history["cooperation_rate"][-10:].mean())
                reward_mean = float(metrics_history["reward_mean"][-10:].mean())
                gini = float(metrics_history["gini"][-10:].mean())
                
                # 학습 곡선 전체 저장 (10% 간격으로 샘플링)
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
                seed_results.append({
                    "seed": seed,
                    "error": str(e),
                    "run_time_sec": run_time,
                })
        
        # SVO 조건별 통계
        valid_results = [r for r in seed_results if "error" not in r]
        if valid_results:
            coop_rates = [r["cooperation_rate"] for r in valid_results]
            rewards = [r["reward_mean"] for r in valid_results]
            ginis = [r["gini"] for r in valid_results]
            
            condition_summary = {
                "seeds": seed_results,
                "mean_cooperation": float(np.mean(coop_rates)),
                "std_cooperation": float(np.std(coop_rates)),
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "mean_gini": float(np.mean(ginis)),
                "n_successful": len(valid_results),
                "n_failed": len(seed_results) - len(valid_results),
            }
        else:
            condition_summary = {
                "seeds": seed_results,
                "n_successful": 0,
                "n_failed": len(seed_results),
                "error": "모든 시드 실패"
            }
        
        results["conditions"][svo_name] = condition_summary
    
    return results


def main():
    """종합 학습 메인 루프."""
    start_time = datetime.now()
    max_duration = timedelta(hours=8)
    
    log.info("=" * 60)
    log.info("EthicaAI Genesis — Phase 1 Full Sweep 시작")
    log.info(f"시작 시각: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"최대 실행 시간: {max_duration}")
    log.info(f"실험 조건: {len(EXPERIMENTS)}개")
    log.info(f"시드: {SEEDS}")
    log.info(f"SVO 조건: {list(SVO_CONDITIONS.keys())}")
    log.info(f"총 학습 횟수: {len(EXPERIMENTS) * len(SEEDS) * len(SVO_CONDITIONS)}회")
    log.info(f"출력 디렉토리: {OUTPUT_DIR}")
    log.info("=" * 60)
    
    all_results = {}
    completed = 0
    total = len(EXPERIMENTS)
    
    for exp_name, exp_def in EXPERIMENTS.items():
        # 시간 제한 체크
        elapsed = datetime.now() - start_time
        if elapsed > max_duration:
            log.warning(f"⏰ 8시간 제한 도달. {completed}/{total} 실험 완료.")
            break
        
        remaining = max_duration - elapsed
        log.info(f"\n📊 진행: {completed}/{total} 완료 | 경과: {elapsed} | 남은 시간: {remaining}")
        
        # 실험별 config 생성
        config = copy.deepcopy(BASE_CONFIG)
        config.update(exp_def["overrides"])
        config["_desc"] = exp_def["desc"]
        
        try:
            result = run_single_experiment(exp_name, config, SEEDS, SVO_CONDITIONS)
            all_results[exp_name] = result
            
            # 실험별 결과 즉시 저장 (중간 저장)
            exp_file = os.path.join(OUTPUT_DIR, f"{exp_name}.json")
            with open(exp_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            log.info(f"💾 저장: {exp_file}")
            
        except Exception as e:
            log.error(f"❌ 실험 {exp_name} 전체 실패: {e}")
            traceback.print_exc()
            all_results[exp_name] = {"error": str(e)}
        
        completed += 1
    
    # 전체 결과 요약 저장
    end_time = datetime.now()
    total_elapsed = end_time - start_time
    
    summary = {
        "meta": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_elapsed_sec": total_elapsed.total_seconds(),
            "total_elapsed_human": str(total_elapsed),
            "platform": _platform,
            "devices": [str(d) for d in jax.devices()],
            "experiments_completed": completed,
            "experiments_total": total,
        },
        "results": all_results,
    }
    
    summary_file = os.path.join(OUTPUT_DIR, "full_sweep_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    log.info("\n" + "=" * 60)
    log.info("🏁 Full Sweep 완료!")
    log.info(f"총 소요: {total_elapsed}")
    log.info(f"완료: {completed}/{total}")
    log.info(f"결과: {summary_file}")
    log.info("=" * 60)
    
    # 최종 비교 테이블 출력
    log.info("\n📊 실험 결과 비교:")
    log.info(f"{'실험':<30} {'Prosocial Coop':>15} {'Indiv. Coop':>15} {'Pro. Reward':>12}")
    log.info("-" * 75)
    
    for exp_name, result in all_results.items():
        if "error" in result:
            log.info(f"{exp_name:<30} {'ERROR':>15}")
            continue
        
        pro = result.get("conditions", {}).get("Prosocial", {})
        ind = result.get("conditions", {}).get("Individualist", {})
        
        pro_coop = f"{pro.get('mean_cooperation', 0):.4f}±{pro.get('std_cooperation', 0):.4f}" if pro.get('n_successful', 0) > 0 else "N/A"
        ind_coop = f"{ind.get('mean_cooperation', 0):.4f}±{ind.get('std_cooperation', 0):.4f}" if ind.get('n_successful', 0) > 0 else "N/A"
        pro_rew = f"{pro.get('mean_reward', 0):.4f}" if pro.get('n_successful', 0) > 0 else "N/A"
        
        log.info(f"{exp_name:<30} {pro_coop:>15} {ind_coop:>15} {pro_rew:>12}")


if __name__ == "__main__":
    main()
