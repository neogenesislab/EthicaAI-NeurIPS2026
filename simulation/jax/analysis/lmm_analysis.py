"""
EthicaAI LMM Analysis Module
선형 혼합 효과 모형(Linear Mixed-Effects Model)을 활용한 통계 분석.
에이전트/시드별 랜덤 효과를 고려하여 OLS보다 엄밀한 추론 수행.

NeurIPS 2026 Workshop 보강용 통계 방법론 (E2).
"""
import numpy as np
import json
from typing import Dict, Any


def sweep_to_dataframe(sweep_results: dict) -> "pd.DataFrame":
    """
    Sweep 결과를 LMM 분석용 DataFrame으로 변환.
    
    각 행 = 하나의 실험 run (SVO × Seed 조합)
    """
    import pandas as pd
    
    rows = []
    for svo_name, svo_data in sweep_results.items():
        theta = svo_data["theta"]
        seeds = svo_data.get("seeds", [])
        
        for i, run in enumerate(svo_data["runs"]):
            seed = seeds[i] if i < len(seeds) else i
            metrics = run["metrics"]
            
            # 최종 epoch 값 사용
            final_reward = metrics["reward_mean"][-1] if "reward_mean" in metrics else 0
            final_coop = metrics["cooperation_rate"][-1] if "cooperation_rate" in metrics else 0
            final_gini = metrics["gini"][-1] if "gini" in metrics else 0
            
            rows.append({
                "svo_name": svo_name,
                "svo_theta": theta,
                "seed": seed,
                "reward": final_reward,
                "cooperation_rate": final_coop,
                "gini": final_gini,
            })
    
    return pd.DataFrame(rows)


def run_lmm_analysis(sweep_results: dict) -> Dict[str, Any]:
    """
    선형 혼합 효과 모형 (LMM) 분석.
    
    고정 효과: SVO theta (연속 처치 변수)
    랜덤 효과: seed (실험 반복 단위)
    
    Model: outcome ~ svo_theta + (1|seed)
    
    Returns:
        Dict with LMM results for each outcome variable
    """
    import pandas as pd
    import statsmodels.formula.api as smf
    
    df = sweep_to_dataframe(sweep_results)
    outcomes = ["reward", "cooperation_rate", "gini"]
    results = {}
    
    for outcome in outcomes:
        try:
            model = smf.mixedlm(
                f"{outcome} ~ svo_theta",
                data=df,
                groups=df["seed"],
            )
            fit = model.fit(reml=True)
            
            results[outcome] = {
                "method": "LMM (REML)",
                "fixed_effect_svo": float(fit.params["svo_theta"]),
                "fixed_effect_intercept": float(fit.params["Intercept"]),
                "se_svo": float(fit.bse["svo_theta"]),
                "p_value": float(fit.pvalues["svo_theta"]),
                "significant": bool(fit.pvalues["svo_theta"] < 0.05),
                "t_stat": float(fit.tvalues["svo_theta"]),
                "aic": float(fit.aic) if hasattr(fit, "aic") else None,
                "bic": float(fit.bic) if hasattr(fit, "bic") else None,
                "random_effect_var": float(fit.cov_re.iloc[0, 0]),
                "n_observations": len(df),
                "n_groups": df["seed"].nunique(),
            }
        except Exception as e:
            results[outcome] = {
                "method": "LMM (REML)",
                "error": str(e),
            }
    
    return results


def print_lmm_report(lmm_results: Dict[str, Any]):
    """LMM 분석 결과 출력."""
    print("\n" + "=" * 60)
    print("Linear Mixed-Effects Model (LMM) Analysis Report")
    print("=" * 60)
    
    for outcome, res in lmm_results.items():
        if "error" in res:
            print(f"\n{outcome}: ERROR - {res['error']}")
            continue
        
        sig = "✅ Significant" if res["significant"] else "❌ Non-significant"
        print(f"\n--- {outcome} ---")
        print(f"  Fixed Effect (SVO θ): {res['fixed_effect_svo']:.6f}")
        print(f"  SE: {res['se_svo']:.6f}")
        print(f"  t-stat: {res['t_stat']:.3f}")
        print(f"  p-value: {res['p_value']:.6f}  [{sig}]")
        print(f"  Random Effect Var (seed): {res['random_effect_var']:.6f}")
        if res.get("aic"):
            print(f"  AIC: {res['aic']:.2f} | BIC: {res['bic']:.2f}")
    
    print("=" * 60)
