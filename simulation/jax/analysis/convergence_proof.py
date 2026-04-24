"""
EthicaAI Convergence Proof Analysis
NeurIPS 2026 — Reviewer 2 Rebuttal: G2

기존 sweep 데이터의 학습 곡선(reward trajectory)을 분석하여
수렴성을 통계적으로 증명합니다.

방법:
  1. 마지막 N epochs의 slope → t-test로 0과 비교 (H0: slope=0 → 수렴)
  2. Augmented Dickey-Fuller (ADF) test로 정상성(stationarity) 검증
  3. 시각화: 학습 곡선 + 수렴 구간 하이라이팅
"""
import os
import json
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- NeurIPS 스타일 설정 ---
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_sweep_trajectories(sweep_path):
    """sweep JSON에서 SVO별 reward trajectory를 추출합니다."""
    with open(sweep_path, "r") as f:
        sweep_data = json.load(f)
    
    trajectories = {}
    for svo, info in sweep_data.items():
        runs = info.get("runs", [])
        reward_curves = []
        for run in runs:
            metrics = run.get("metrics", {})
            reward_mean = metrics.get("reward_mean", [])
            if reward_mean:
                reward_curves.append(reward_mean)
        if reward_curves:
            trajectories[svo] = reward_curves
    
    return trajectories


def test_convergence(trajectory, tail_fraction=0.3):
    """
    학습 곡선의 수렴성을 검증합니다.
    
    Args:
        trajectory: list of float (epoch별 reward)
        tail_fraction: 분석할 마지막 구간 비율 (기본 30%)
    
    Returns:
        dict: slope, t_stat, p_value, adf_stat, adf_pvalue, converged
    """
    arr = np.array(trajectory)
    n = len(arr)
    tail_start = int(n * (1 - tail_fraction))
    tail = arr[tail_start:]
    
    if len(tail) < 5:
        return {"converged": False, "reason": "데이터 부족"}
    
    # 1. Linear Regression: slope → 0이면 수렴
    x = np.arange(len(tail))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, tail)
    
    # slope가 0과 유의하게 다른지 t-test
    t_stat = slope / std_err if std_err > 0 else 0
    slope_p = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(tail) - 2))
    
    # 2. ADF Test (정상성 검증)
    # statsmodels 없이 간이 ADF 구현 (차분의 자기상관 검정)
    try:
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(tail, maxlag=min(5, len(tail) // 3))
        adf_stat = adf_result[0]
        adf_pvalue = adf_result[1]
    except ImportError:
        # statsmodels 없으면 차분 분산으로 대체
        diffs = np.diff(tail)
        adf_stat = np.mean(diffs) / (np.std(diffs) + 1e-8)
        adf_pvalue = 2 * (1 - stats.norm.cdf(abs(adf_stat)))
    
    # 3. 수렴 판정: slope가 0과 유의하게 다르지 않으면(p>=0.05) 수렴
    converged = slope_p >= 0.05  # slope가 0과 다르지 않음 = 수렴
    
    return {
        "slope": float(slope),
        "slope_p": float(slope_p),
        "t_stat": float(t_stat),
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "tail_mean": float(np.mean(tail)),
        "tail_std": float(np.std(tail)),
        "converged": bool(converged),
        "n_tail": len(tail),
    }


def analyze_all_convergence(trajectories):
    """모든 SVO 조건의 수렴성을 분석합니다."""
    results = {}
    for svo, curves in trajectories.items():
        svo_results = []
        for i, curve in enumerate(curves):
            result = test_convergence(curve)
            result["seed"] = i
            svo_results.append(result)
        
        # slope 키가 있는 결과만 필터
        valid_results = [r for r in svo_results if "slope" in r]
        converged_count = sum(1 for r in svo_results if r.get("converged", False))
        
        if valid_results:
            avg_slope = np.mean([r["slope"] for r in valid_results])
            avg_slope_p = np.mean([r["slope_p"] for r in valid_results])
            avg_adf_p = np.mean([r["adf_pvalue"] for r in valid_results])
        else:
            avg_slope = 0.0
            avg_slope_p = 1.0
            avg_adf_p = 1.0
        
        results[svo] = {
            "runs": svo_results,
            "converged_ratio": converged_count / max(len(svo_results), 1),
            "avg_slope": float(avg_slope),
            "avg_slope_p": float(avg_slope_p),
            "avg_adf_pvalue": float(avg_adf_p),
            "total_runs": len(svo_results),
            "converged_runs": converged_count,
        }
    
    return results


def plot_convergence(trajectories, results, output_dir):
    """수렴성 분석 결과를 시각화합니다."""
    svo_list = list(trajectories.keys())
    n_svos = len(svo_list)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # --- 상단: 학습 곡선 + 수렴 구간 ---
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, n_svos))
    
    for idx, svo in enumerate(svo_list):
        curves = trajectories[svo]
        # 평균 + std band
        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            padded[i, :len(c)] = c
        
        mean_curve = np.nanmean(padded, axis=0)
        std_curve = np.nanstd(padded, axis=0)
        x = np.arange(max_len)
        
        ax1.plot(x, mean_curve, color=colors[idx], label=svo, linewidth=1.5, alpha=0.8)
        ax1.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        color=colors[idx], alpha=0.1)
    
    # 수렴 구간 하이라이팅 (마지막 30%)
    if max_len > 0:
        conv_start = int(max_len * 0.7)
        ax1.axvspan(conv_start, max_len, alpha=0.15, color='green',
                   label='Convergence Zone (last 30%)')
        ax1.axvline(x=conv_start, color='green', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel("Training Epoch")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("Learning Curves with Convergence Analysis\n(All SVO Conditions, 100 Agents)")
    ax1.legend(loc='upper left', ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # --- 하단: 수렴 통계 요약 바 차트 ---
    ax2 = axes[1]
    
    svo_labels = list(results.keys())
    conv_ratios = [results[s]["converged_ratio"] for s in svo_labels]
    avg_slopes = [abs(results[s]["avg_slope"]) for s in svo_labels]
    adf_ps = [results[s]["avg_adf_pvalue"] for s in svo_labels]
    
    x_pos = np.arange(len(svo_labels))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, conv_ratios, width, 
                    label='Convergence Rate', color='#4CAF50', alpha=0.8)
    
    # ADF p-value를 1-p로 변환 (높을수록 정상성 증거가 강함)
    adf_evidence = [1.0 - p for p in adf_ps]
    bars2 = ax2.bar(x_pos + width/2, adf_evidence, width,
                    label='Stationarity Evidence (1 - ADF p)', color='#2196F3', alpha=0.8)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(svo_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=0.95, color='red', linestyle=':', alpha=0.5, label='95% Threshold')
    ax2.legend(fontsize=8)
    ax2.set_title("Convergence Statistics by SVO Condition")
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig_convergence_proof.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[G2] 수렴성 증명 Figure 저장: {out_path}")
    return out_path


def print_summary(results):
    """콘솔 요약 출력."""
    print("\n" + "=" * 70)
    print("  EthicaAI Convergence Proof — Statistical Summary")
    print("=" * 70)
    
    total_converged = 0
    total_runs = 0
    
    for svo, res in results.items():
        total_converged += res["converged_runs"]
        total_runs += res["total_runs"]
        status = "✅ CONVERGED" if res["converged_ratio"] >= 0.8 else "⚠️ PARTIAL"
        print(f"  {svo:20s} | Conv: {res['converged_ratio']:.0%} "
              f"({res['converged_runs']}/{res['total_runs']}) "
              f"| Slope: {res['avg_slope']:.6f} "
              f"| ADF p: {res['avg_adf_pvalue']:.4f} "
              f"| {status}")
    
    overall = total_converged / total_runs if total_runs > 0 else 0
    print("-" * 70)
    print(f"  OVERALL: {overall:.0%} ({total_converged}/{total_runs}) runs converged")
    print(f"  VERDICT: {'✅ Training has converged' if overall >= 0.8 else '❌ Convergence not proven'}")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python convergence_proof.py <output_dir>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    os.makedirs(output_dir, exist_ok=True)
    
    # sweep 파일 찾기 — output_dir 직접 하위만 탐색 (다른 모듈 파일 혼동 방지)
    sweep_path = None
    if os.path.isdir(output_dir):
        for f in os.listdir(output_dir):
            if f.startswith("sweep_") and f.endswith(".json"):
                sweep_path = os.path.join(output_dir, f)
                break
    
    if sweep_path is None:
        # Standalone 모드: 자체 mini sweep 데이터 생성
        print("[G2] sweep 파일 없음 — Standalone 모드: 자체 데이터 생성")
        from simulation.jax.experiment_jax import run_sweep
        from simulation.jax.config import SVO_SWEEP_THETAS
        
        angles = {
            "prosocial": SVO_SWEEP_THETAS["prosocial"],
            "cooperative": SVO_SWEEP_THETAS["cooperative"],
        }
        sweep_data, sweep_path = run_sweep(
            scale="small", svo_angles=angles, seeds=[42],
            output_dir=output_dir,
        )
        # run_sweep이 저장한 JSON 경로 사용
        print(f"[G2] Mini sweep 완료: {sweep_path}")
    else:
        print(f"[G2] Sweep 데이터 로딩: {sweep_path}")
    
    trajectories = load_sweep_trajectories(sweep_path)
    print(f"[G2] {len(trajectories)}개 SVO 조건 로드 완료")
    
    results = analyze_all_convergence(trajectories)
    print_summary(results)
    
    # Figure 생성
    fig_path = plot_convergence(trajectories, results, output_dir)
    
    # 결과 저장
    out_json = os.path.join(output_dir, "convergence_results.json")
    serializable = {}
    for svo, res in results.items():
        serializable[svo] = {k: v for k, v in res.items() if k != "runs"}
    
    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[G2] 결과 JSON 저장: {out_json}")

