"""
EthicaAI Re-analysis Script (NeurIPS 2026 Enhanced)
기존 Sweep 데이터를 로드하여 다층 통계분석을 수행:
  1. OLS + HAC Robust SE (기존)
  2. Linear Mixed-Effects Model (LMM) (신규)
  3. Bootstrap Confidence Interval (신규)
  4. 스케일 비교 분석 (20 vs 100 에이전트) (신규)
"""
import os
import sys
import json
import numpy as np

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import causal_jax, paper_figures

# LMM/Bootstrap은 선택적 임포트 (의존성 없으면 건너뜀)
try:
    from analysis.lmm_analysis import run_lmm_analysis, print_lmm_report
    HAS_LMM = True
except ImportError:
    HAS_LMM = False

try:
    from analysis.bootstrap_ci import run_bootstrap_analysis, print_bootstrap_report
    HAS_BOOTSTRAP = True
except ImportError:
    HAS_BOOTSTRAP = False


def find_latest_run(base_dir, prefix="run_large_"):
    """최신 실험 디렉토리 자동 탐색."""
    if not os.path.exists(base_dir):
        return None
    runs = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
    if not runs:
        return None
    return os.path.join(base_dir, sorted(runs)[-1])


def load_sweep_data(run_dir):
    """Sweep 결과 JSON 로드."""
    sweep_files = [f for f in os.listdir(run_dir) if f.startswith("sweep_") and f.endswith(".json")]
    if not sweep_files:
        print(f"No sweep file found in {run_dir}")
        return None
    sweep_path = os.path.join(run_dir, sweep_files[0])
    with open(sweep_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_full_analysis(sweep_results, run_dir):
    """
    전체 분석 파이프라인 실행.
    OLS(HAC) + LMM + Bootstrap + Figure 생성.
    """
    fig_dir = os.path.join(run_dir, "figures")
    table_dir = os.path.join(run_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    
    all_results = {}
    
    # === 1. OLS + HAC Robust SE ===
    print("\n[1/4] OLS Causal Analysis (HAC Robust SE)...")
    causal_results = causal_jax.run_causal_analysis(sweep_results)
    causal_jax.print_causal_report(causal_results)
    all_results["ols_hac"] = causal_results
    
    causal_path = os.path.join(run_dir, "causal_results.json")
    with open(causal_path, "w", encoding="utf-8") as f:
        json.dump(causal_results, f, indent=2, ensure_ascii=False)
    
    # === 2. LMM ===
    if HAS_LMM:
        print("\n[2/4] Linear Mixed-Effects Model (LMM)...")
        try:
            lmm_results = run_lmm_analysis(sweep_results)
            print_lmm_report(lmm_results)
            all_results["lmm"] = lmm_results
            
            lmm_path = os.path.join(run_dir, "lmm_results.json")
            with open(lmm_path, "w", encoding="utf-8") as f:
                json.dump(lmm_results, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"  LMM failed: {e}")
    else:
        print("\n[2/4] LMM skipped (statsmodels not available)")
    
    # === 3. Bootstrap CI ===
    if HAS_BOOTSTRAP:
        print("\n[3/4] Bootstrap Confidence Intervals...")
        try:
            bootstrap_results = run_bootstrap_analysis(sweep_results)
            print_bootstrap_report(bootstrap_results)
            all_results["bootstrap"] = bootstrap_results
            
            boot_path = os.path.join(run_dir, "bootstrap_results.json")
            with open(boot_path, "w", encoding="utf-8") as f:
                json.dump(bootstrap_results, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"  Bootstrap failed: {e}")
    else:
        print("\n[3/4] Bootstrap skipped")
    
    # === 4. Figure 생성 ===
    print("\n[4/4] Generating Figures...")
    paper_figures.plot_causal_forest(causal_results, fig_dir)
    paper_figures.plot_role_specialization(sweep_results, fig_dir)
    paper_figures.plot_summary_heatmap(sweep_results, fig_dir)
    paper_figures.generate_causal_table(causal_results, table_dir)
    
    # === 통합 리포트 ===
    print_comparison_report(all_results)
    
    # 전체 결과 저장
    report_path = os.path.join(run_dir, "full_analysis_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull report saved to {report_path}")
    
    return all_results


def print_comparison_report(all_results):
    """OLS vs LMM vs Bootstrap 비교 리포트 출력."""
    print("\n" + "=" * 70)
    print("STATISTICAL METHOD COMPARISON (NeurIPS 2026)")
    print("=" * 70)
    print(f"{'Outcome':<20} {'OLS(HAC) p':<14} {'LMM p':<14} {'Bootstrap CI':<24} {'Agree?':<8}")
    print("-" * 70)
    
    ols = all_results.get("ols_hac", {})
    lmm = all_results.get("lmm", {})
    boot = all_results.get("bootstrap", {})
    
    outcome_map = {
        "reward": ("H1_reward", "reward", "reward"),
        "cooperation": ("H2_cooperation", "cooperation_rate", "cooperation_rate"),
        "gini": ("H3_gini", "gini", "gini"),
    }
    
    for label, (ols_key, lmm_key, boot_key) in outcome_map.items():
        ols_p = ols.get(ols_key, {}).get("p_value", "N/A")
        lmm_p = lmm.get(lmm_key, {}).get("p_value", "N/A")
        boot_ci = boot.get(boot_key, {})
        
        ols_str = f"{ols_p:.4f}" if isinstance(ols_p, float) else str(ols_p)
        lmm_str = f"{lmm_p:.4f}" if isinstance(lmm_p, float) else str(lmm_p)
        
        if boot_ci:
            boot_str = f"[{boot_ci.get('ci_lower', 0):.4f}, {boot_ci.get('ci_upper', 0):.4f}]"
        else:
            boot_str = "N/A"
        
        # 일치 여부 확인
        ols_sig = ols.get(ols_key, {}).get("significant", None)
        lmm_sig = lmm.get(lmm_key, {}).get("significant", None)
        boot_sig = boot.get(boot_key, {}).get("significant", None)
        
        sigs = [s for s in [ols_sig, lmm_sig, boot_sig] if s is not None]
        agree = "✅" if len(set(sigs)) <= 1 and sigs else "⚠️"
        
        print(f"{label:<20} {ols_str:<14} {lmm_str:<14} {boot_str:<24} {agree:<8}")
    
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        run_dir = find_latest_run("simulation/outputs")
        if not run_dir:
            print("No run found. Specify a run directory.")
            sys.exit(1)
        print(f"Auto-selected: {run_dir}")
    else:
        run_dir = sys.argv[1]

    sweep_results = load_sweep_data(run_dir)
    if not sweep_results:
        sys.exit(1)
    
    print(f"Loaded sweep data from {run_dir}")
    print(f"SVO conditions: {len(sweep_results)}")
    print(f"Methods: OLS(HAC){' + LMM' if HAS_LMM else ''}{' + Bootstrap' if HAS_BOOTSTRAP else ''}")
    
    run_full_analysis(sweep_results, run_dir)
    print("\nRe-analysis Complete.")


if __name__ == "__main__":
    main()
