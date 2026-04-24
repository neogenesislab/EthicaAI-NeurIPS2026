"""
EthicaAI Bootstrap CI Module
비모수적 부트스트랩을 통한 ATE 신뢰구간 추정.
OLS/LMM의 정규성 가정에 의존하지 않는 강건한 추론.

NeurIPS 2026 Workshop 보강용 통계 방법론 (E2).
"""
import numpy as np
from typing import Dict, Any, Optional


def bootstrap_ate(
    T: np.ndarray,
    Y: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    비모수적 부트스트랩 ATE 신뢰구간 추정.
    
    Args:
        T: 처치 변수 (SVO theta)
        Y: 결과 변수 (reward/cooperation/gini)
        n_bootstrap: 부트스트랩 반복 횟수
        ci_level: 신뢰구간 수준 (default: 95%)
        seed: 재현성을 위한 랜덤 시드
        
    Returns:
        Dict with ATE estimate, CI, and bootstrap distribution stats
    """
    rng = np.random.default_rng(seed)
    n = len(T)
    alpha = 1 - ci_level
    
    estimates = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        T_boot = T[idx]
        Y_boot = Y[idx]
        
        # OLS ATE 추정 (간단 버전)
        X = np.column_stack([np.ones(n), T_boot])
        try:
            beta = np.linalg.lstsq(X, Y_boot, rcond=None)[0]
            estimates[i] = beta[1]  # SVO theta 계수 = ATE
        except np.linalg.LinAlgError:
            estimates[i] = np.nan
    
    # NaN 제거
    valid = estimates[~np.isnan(estimates)]
    
    return {
        "method": f"Bootstrap ({n_bootstrap} iterations)",
        "ate_mean": float(np.mean(valid)),
        "ate_median": float(np.median(valid)),
        "ate_std": float(np.std(valid)),
        "ci_lower": float(np.percentile(valid, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(valid, 100 * (1 - alpha / 2))),
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
        "n_valid": len(valid),
        "significant": bool(
            np.percentile(valid, 100 * alpha / 2) * np.percentile(valid, 100 * (1 - alpha / 2)) > 0
        ),  # CI가 0을 포함하지 않으면 유의
    }


def run_bootstrap_analysis(sweep_results: dict) -> Dict[str, Any]:
    """
    전체 Sweep 결과에 대해 부트스트랩 분석 수행.
    
    Args:
        sweep_results: Sweep 실험 결과 dict
        
    Returns:
        Dict with bootstrap CI for each outcome variable
    """
    # 데이터 추출
    thetas = []
    rewards = []
    coops = []
    ginis = []
    
    for svo_name, svo_data in sweep_results.items():
        theta = svo_data["theta"]
        for run in svo_data["runs"]:
            metrics = run["metrics"]
            thetas.append(theta)
            rewards.append(metrics["reward_mean"][-1] if "reward_mean" in metrics else 0)
            coops.append(metrics["cooperation_rate"][-1] if "cooperation_rate" in metrics else 0)
            ginis.append(metrics["gini"][-1] if "gini" in metrics else 0)
    
    T = np.array(thetas)
    
    results = {
        "reward": bootstrap_ate(T, np.array(rewards)),
        "cooperation_rate": bootstrap_ate(T, np.array(coops)),
        "gini": bootstrap_ate(T, np.array(ginis)),
    }
    
    return results


def print_bootstrap_report(bootstrap_results: Dict[str, Any]):
    """부트스트랩 분석 결과 출력."""
    print("\n" + "=" * 60)
    print("Bootstrap Confidence Interval Analysis Report")
    print("=" * 60)
    
    for outcome, res in bootstrap_results.items():
        sig = "✅ Significant" if res["significant"] else "❌ Non-significant"
        print(f"\n--- {outcome} ---")
        print(f"  ATE (mean): {res['ate_mean']:.6f}")
        print(f"  ATE (median): {res['ate_median']:.6f}")
        print(f"  95% CI: [{res['ci_lower']:.6f}, {res['ci_upper']:.6f}]")
        print(f"  Bootstrap SE: {res['ate_std']:.6f}")
        print(f"  {sig} (CI {'excludes' if res['significant'] else 'includes'} zero)")
    
    print("=" * 60)
