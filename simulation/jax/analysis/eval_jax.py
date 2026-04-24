"""
EthicaAI JAX Evaluation Module
Metrics calculation for research-grade analysis.
"""
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List


def gini_coefficient(rewards: np.ndarray) -> float:
    """
    Gini 계수 계산 (보상 불평등도).
    0 = 완전 평등, 1 = 완전 불평등.
    
    G = (2 * Σ(i * y_sorted_i)) / (n * Σy) - (n+1)/n
    """
    rewards = np.array(rewards, dtype=np.float64)
    if len(rewards) < 2 or np.sum(np.abs(rewards)) < 1e-10:
        return 0.0
    
    # Shift to non-negative
    shifted = rewards - rewards.min() + 1e-8
    n = len(shifted)
    sorted_r = np.sort(shifted)
    index = np.arange(1, n + 1)
    
    return float((2.0 * np.sum(index * sorted_r)) / (n * np.sum(sorted_r)) - (n + 1.0) / n)


def cooperation_rate(actions: np.ndarray, beam_action: int = 7) -> float:
    """
    협력률 계산 (청소 행동 = BEAM 비율).
    """
    total = actions.size
    if total == 0:
        return 0.0
    return float(np.sum(actions == beam_action) / total)


def sustainability_index(initial_apples: float, final_apples: float) -> float:
    """
    지속가능성 지수: 자원 유지율.
    SI = final / initial (1.0 = 완전 유지, 0 = 고갈)
    """
    if initial_apples < 1e-8:
        return 0.0
    return float(min(final_apples / initial_apples, 1.0))


def specialization_index(thresholds: np.ndarray) -> float:
    """
    전문화 지수: 역치 분산 기반.
    높을수록 에이전트 간 역할 분화가 뚜렷함.
    
    thresholds: (N, Tasks) 형태
    """
    if thresholds.shape[0] < 2:
        return 0.0
    
    # 각 Task별 에이전트간 역치 분산의 평균
    var_per_task = np.var(thresholds, axis=0)
    return float(np.mean(var_per_task))


def social_welfare(rewards: np.ndarray) -> float:
    """사회적 복지: 총 보상 합."""
    return float(np.sum(rewards))


def fairness_ratio(rewards: np.ndarray) -> float:
    """
    공정성 비율: min/max.
    1.0 = 완전 공정, 0 = 극단적 불공정.
    """
    r_max = np.max(rewards)
    r_min = np.min(rewards)
    if abs(r_max) < 1e-8:
        return 1.0
    return float(r_min / r_max) if r_max > 0 else 0.0


def hypothesis_test(group_a: np.ndarray, group_b: np.ndarray,
                     method: str = "welch_t") -> Dict[str, float]:
    """
    두 그룹 간 통계적 유의성 검정.
    
    Returns:
        t_stat, p_value, cohens_d (효과 크기)
    """
    from scipy import stats
    
    if len(group_a) < 2 or len(group_b) < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0}
    
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    
    # Cohen's d (효과 크기)
    pooled_std = np.sqrt(
        (np.var(group_a) * (len(group_a)-1) + np.var(group_b) * (len(group_b)-1))
        / (len(group_a) + len(group_b) - 2)
    )
    cohens_d = (np.mean(group_a) - np.mean(group_b)) / (pooled_std + 1e-8)
    
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant": bool(p_value < 0.05),
    }


def evaluate_sweep(sweep_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    전체 Sweep 결과를 종합 평가.
    
    Args:
        sweep_results: run_sweep()의 반환값
    
    Returns:
        condition별 집계 메트릭 + 조건 간 비교 통계
    """
    eval_output = {}
    
    # 조건별 집계
    for name, data in sweep_results.items():
        runs = data["runs"]
        
        final_rewards = [r["metrics"]["reward_mean"][-1] for r in runs]
        final_coops = [r["metrics"]["cooperation_rate"][-1] for r in runs]
        final_ginis = [r["metrics"]["gini"][-1] for r in runs]
        
        eval_output[name] = {
            "theta": data["theta"],
            "reward": {"mean": float(np.mean(final_rewards)), "std": float(np.std(final_rewards))},
            "cooperation": {"mean": float(np.mean(final_coops)), "std": float(np.std(final_coops))},
            "gini": {"mean": float(np.mean(final_ginis)), "std": float(np.std(final_ginis))},
            "threshold_clean": {"mean": float(np.mean([r["final_thresholds_mean"]["clean"] for r in runs]))},
            "threshold_harvest": {"mean": float(np.mean([r["final_thresholds_mean"]["harvest"] for r in runs]))},
        }
    
    # 조건 간 비교 (이기적 vs 친사회적)
    conditions = list(sweep_results.keys())
    if len(conditions) >= 2:
        first = sweep_results[conditions[0]]["runs"]
        last = sweep_results[conditions[-1]]["runs"]
        
        rewards_first = np.array([r["metrics"]["reward_mean"][-1] for r in first])
        rewards_last = np.array([r["metrics"]["reward_mean"][-1] for r in last])
        
        eval_output["comparison"] = {
            "groups": f"{conditions[0]} vs {conditions[-1]}",
            "test": hypothesis_test(rewards_first, rewards_last),
        }
    
    return eval_output
