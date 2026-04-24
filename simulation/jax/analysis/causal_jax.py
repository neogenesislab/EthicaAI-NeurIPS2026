"""
EthicaAI JAX Causal Inference Module
ATE estimation for treatment effect analysis.
"""
import numpy as np
from typing import Dict, Any, List, Tuple


def estimate_ate_ols(T: np.ndarray, Y: np.ndarray, 
                      X: np.ndarray = None) -> Dict[str, float]:
    """
    OLS(Ordinary Least Squares) 기반 ATE(Average Treatment Effect) 추정.
    
    Y = α + β·T + γ·X + ε
    β가 Treatment Effect (ATE)
    
    Args:
        T: Treatment 변수 (예: SVO 각도), shape (n,)
        Y: Outcome 변수 (예: 총 보상), shape (n,)
        X: Confounder 변수 (예: 시드), shape (n, d) or None
    
    Returns:
        ate, se, t_stat, p_value, r_squared
    """
    n = len(T)
    if n < 3:
        return {"ate": 0.0, "se": float("inf"), "t_stat": 0.0, 
                "p_value": 1.0, "r_squared": 0.0, "method": "ols"}
    
    # Design Matrix: [1, T, X]
    ones = np.ones((n, 1))
    T_col = T.reshape(-1, 1)
    
    if X is not None and X.ndim == 2:
        design = np.hstack([ones, T_col, X])
    else:
        design = np.hstack([ones, T_col])
    
    # OLS: β = (X'X)^-1 X'Y
    try:
        XtX_inv = np.linalg.inv(design.T @ design)
        beta = XtX_inv @ (design.T @ Y)
    except np.linalg.LinAlgError:
        return {"ate": 0.0, "se": float("inf"), "t_stat": 0.0, 
                "p_value": 1.0, "r_squared": 0.0, "method": "ols"}
    
    # Residuals
    Y_pred = design @ beta
    residuals = Y - Y_pred
    
    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_squared = 1 - ss_res / (ss_tot + 1e-10)
    
    # Standard Errors (Basic OLS)
    dof = n - design.shape[1]
    mse = ss_res / max(dof, 1)
    se_beta_ols = np.sqrt(np.diag(XtX_inv) * mse)
    
    # Robust Standard Errors (HC1) - Heteroscedasticity Consistent
    # Var(beta) = (X'X)^-1 * (X' * diag(res^2) * X) * (X'X)^-1 * (n / (n-k))
    resid_sq_diag = np.diag(residuals.flatten() ** 2)
    HC1_factor = n / max(dof, 1)
    meat = design.T @ resid_sq_diag @ design
    sandwich = XtX_inv @ meat @ XtX_inv * HC1_factor
    se_beta_robust = np.sqrt(np.diag(sandwich))
    
    # Use Robust SE for inference
    se_beta = se_beta_robust
    
    # ATE is beta[1] (coefficient of T)
    ate = float(beta[1])
    se = float(se_beta[1])
    t_stat = ate / (se + 1e-10)
    
    # p-value (two-tailed)
    from scipy import stats
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), dof)))
    
    return {
        "ate": ate,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "r_squared": r_squared,
        "method": "ols",
        "significant": bool(p_value < 0.05),
    }


def run_causal_analysis(sweep_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sweep 결과에서 인과분석 실행.
    
    가설:
        H1: SVO θ → 총 보상 (역U자 곡선 가능)
        H2: SVO θ → 협력률
        H3: 역치 분산 → 사회적 복지
    """
    # Collect data across all conditions and seeds
    T_svo = []     # Treatment: SVO angle
    Y_reward = []  # Outcome 1: Final reward
    Y_coop = []    # Outcome 2: Final cooperation rate
    Y_gini = []    # Outcome 3: Final Gini coefficient
    X_seed = []    # Confounder: seed index
    
    for name, data in sweep_results.items():
        theta = data["theta"]
        for i, run in enumerate(data["runs"]):
            T_svo.append(theta)
            Y_reward.append(run["metrics"]["reward_mean"][-1])
            Y_coop.append(run["metrics"]["cooperation_rate"][-1])
            Y_gini.append(run["metrics"]["gini"][-1])
            X_seed.append(i)
    
    T_svo = np.array(T_svo)
    Y_reward = np.array(Y_reward)
    Y_coop = np.array(Y_coop)
    Y_gini = np.array(Y_gini)
    X_seed = np.array(X_seed).reshape(-1, 1)
    
    results = {}
    
    # H1: SVO → Reward
    results["H1_svo_reward"] = estimate_ate_ols(T_svo, Y_reward, X_seed)
    results["H1_svo_reward"]["hypothesis"] = "SVO 각도 증가 → 보상 변화"
    
    # 효과 크기 (Cohen's f²) 추가
    r2 = results["H1_svo_reward"].get("r_squared", 0)
    f2 = r2 / (1 - r2 + 1e-10)
    results["H1_svo_reward"]["cohens_f2"] = float(f2)
    results["H1_svo_reward"]["effect_label"] = (
        "large" if f2 >= 0.35 else "medium" if f2 >= 0.15 else "small"
    )
    
    # H2: SVO → Cooperation
    results["H2_svo_cooperation"] = estimate_ate_ols(T_svo, Y_coop, X_seed)
    results["H2_svo_cooperation"]["hypothesis"] = "SVO 각도 증가 → 협력률 변화"
    
    # H3: SVO → Gini (불평등도)
    results["H3_svo_gini"] = estimate_ate_ols(T_svo, Y_gini, X_seed)
    results["H3_svo_gini"]["hypothesis"] = "SVO 각도 증가 → 불평등도 변화"
    
    # H3 효과 크기
    r2_g = results["H3_svo_gini"].get("r_squared", 0)
    f2_g = r2_g / (1 - r2_g + 1e-10)
    results["H3_svo_gini"]["cohens_f2"] = float(f2_g)
    results["H3_svo_gini"]["effect_label"] = (
        "large" if f2_g >= 0.35 else "medium" if f2_g >= 0.15 else "small"
    )
    
    # H1b: Quadratic (역U자) — SVO + SVO² → Reward
    T_quad = np.column_stack([T_svo, T_svo ** 2])
    ones = np.ones((len(T_svo), 1))
    
    if X_seed is not None:
        design_quad = np.hstack([ones, T_quad, X_seed])
    else:
        design_quad = np.hstack([ones, T_quad])
    
    try:
        XtX_inv = np.linalg.inv(design_quad.T @ design_quad)
        beta_q = XtX_inv @ (design_quad.T @ Y_reward)
        results["H1b_quadratic"] = {
            "beta_linear": float(beta_q[1]),
            "beta_quadratic": float(beta_q[2]),
            "inverted_u": bool(beta_q[2] < 0),  # 역U자 여부
            "optimal_theta": float(-beta_q[1] / (2 * beta_q[2] + 1e-10)) if beta_q[2] < 0 else None,
            "method": "ols_quadratic",
            "hypothesis": "SVO와 보상의 역U자 관계",
        }
    except np.linalg.LinAlgError:
        results["H1b_quadratic"] = {"error": "singular_matrix"}
    
    # 단조성 검정 (B1)
    results["monotonicity"] = monotonicity_tests(sweep_results)
    
    return results


def monotonicity_tests(sweep_results: Dict[str, Any]) -> Dict[str, Any]:
    """단조성 통계 검정: Spearman 순위상관 + Kruskal-Wallis."""
    from scipy.stats import spearmanr, kruskal
    
    thetas, rewards, ginis, coops = [], [], [], []
    groups_reward = []  # 그룹별 보상 리스트
    groups_gini = []    # 그룹별 Gini 리스트
    
    for name, data in sweep_results.items():
        group_r, group_g = [], []
        for run in data["runs"]:
            thetas.append(data["theta"])
            r_val = run["metrics"]["reward_mean"][-1]
            g_val = run["metrics"]["gini"][-1]
            c_val = run["metrics"]["cooperation_rate"][-1]
            rewards.append(r_val)
            ginis.append(g_val)
            coops.append(c_val)
            group_r.append(r_val)
            group_g.append(g_val)
        groups_reward.append(group_r)
        groups_gini.append(group_g)
    
    results = {}
    
    # Spearman 순위상관 (단조 관계 검정)
    for metric_name, Y in [("reward", rewards), ("gini", ginis), ("coop", coops)]:
        rho, p = spearmanr(thetas, Y)
        results[f"spearman_{metric_name}"] = {
            "rho": float(rho), "p": float(p),
            "significant": bool(p < 0.05),
            "direction": "negative" if rho < 0 else "positive"
        }
    
    # Kruskal-Wallis (그룹 간 차이 검정, 비모수)
    if len(groups_reward) >= 2:
        H_r, p_kw_r = kruskal(*groups_reward)
        results["kruskal_reward"] = {"H": float(H_r), "p": float(p_kw_r), "significant": bool(p_kw_r < 0.05)}
    if len(groups_gini) >= 2:
        H_g, p_kw_g = kruskal(*groups_gini)
        results["kruskal_gini"] = {"H": float(H_g), "p": float(p_kw_g), "significant": bool(p_kw_g < 0.05)}
    
    return results


def print_causal_report(causal_results: Dict[str, Any]):
    """인과분석 결과를 보기 좋게 출력."""
    print("\n" + "=" * 60)
    print("CAUSAL ANALYSIS REPORT")
    print("=" * 60)
    
    for name, result in causal_results.items():
        if name == "monotonicity":
            print(f"\n--- Monotonicity Tests ---")
            for k, v in result.items():
                if "spearman" in k:
                    sig = "Y" if v.get("significant") else "N"
                    print(f"  {k}: rho={v['rho']:.4f}, p={v['p']:.6f} [{sig}] ({v['direction']})")
                elif "kruskal" in k:
                    sig = "Y" if v.get("significant") else "N"
                    print(f"  {k}: H={v['H']:.2f}, p={v['p']:.6f} [{sig}]")
            continue
        
        print(f"\n--- {name} ---")
        if "hypothesis" in result:
            print(f"  가설: {result['hypothesis']}")
        if "ate" in result:
            sig = "Y" if result.get("significant") else "N"
            print(f"  ATE = {result['ate']:.6f} (SE={result['se']:.6f})")
            print(f"  t={result['t_stat']:.3f}, p={result['p_value']:.4f} [{sig}]")
            print(f"  R² = {result['r_squared']:.4f}")
            if "cohens_f2" in result:
                print(f"  Cohen's f² = {result['cohens_f2']:.4f} ({result.get('effect_label', '')})")
        if "inverted_u" in result:
            print(f"  beta_linear={result['beta_linear']:.6f}, beta_quad={result['beta_quadratic']:.6f}")
            print(f"  역U자: {'예' if result['inverted_u'] else '아니오'}")
            if result.get("optimal_theta"):
                print(f"  최적 theta = {result['optimal_theta']:.4f} rad")
    
    print("=" * 60)
