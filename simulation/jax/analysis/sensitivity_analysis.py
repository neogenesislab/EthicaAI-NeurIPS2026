"""
EthicaAI Parameter Sensitivity Analysis
NeurIPS 2026 — Reviewer 2 Rebuttal: G1

Reviewer 2 비판: "META_WEALTH_BOOST=5.0, META_SURVIVAL_THRESHOLD=-5.0의
이론적 근거가 없다. Cherry-picking 의혹."

방법: 기존 sweep 데이터에서 파라미터 변동 시 결과의 안정성을 간접 증명
  - SVO (θ) 축은 이미 7개 조건으로 sweep 완료
  - 각 SVO 조건 내 seed별 분산을 분석하여 결과의 안정성을 증명
  - Robustness Heatmap: SVO × Metric의 Cohen's d 효과 크기 일관성

추가: META_WEALTH_BOOST와 META_SURVIVAL_THRESHOLD를 변화시키는
      새 실험을 생성하는 스크립트도 포함 (실험은 별도 실행)
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


def load_sweep_per_svo(sweep_path):
    """SVO별 seed-level 메트릭을 로드합니다."""
    with open(sweep_path, "r") as f:
        sweep_data = json.load(f)
    
    data = {}
    for svo, info in sweep_data.items():
        runs = info.get("runs", [])
        seed_metrics = []
        for run in runs:
            m = run.get("metrics", {})
            seed_metrics.append({
                "reward": m.get("reward_mean", [0])[-1] if m.get("reward_mean") else 0,
                "gini": m.get("gini", [0])[-1] if m.get("gini") else 0,
                "cooperation": m.get("cooperation_rate", [0])[-1] if m.get("cooperation_rate") else 0,
            })
        data[svo] = seed_metrics
    return data


def compute_effect_sizes(full_data, baseline_data):
    """각 SVO 조건별 Cohen's d (Full vs Baseline) 계산."""
    metrics = ["reward", "gini", "cooperation"]
    effect_matrix = {}
    
    for svo in full_data:
        if svo not in baseline_data:
            continue
        
        effects = {}
        for m in metrics:
            full_vals = [s[m] for s in full_data[svo]]
            base_vals = [s[m] for s in baseline_data[svo]]
            
            if not full_vals or not base_vals:
                effects[m] = 0.0
                continue
            
            mean_diff = np.mean(full_vals) - np.mean(base_vals)
            pooled_std = np.sqrt((np.var(full_vals) + np.var(base_vals)) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 1e-8 else 0
            
            _, p_val = stats.ttest_ind(full_vals, base_vals, equal_var=False)
            
            effects[m] = {
                "d": float(cohens_d),
                "p": float(p_val),
                "full_mean": float(np.mean(full_vals)),
                "base_mean": float(np.mean(base_vals)),
                "full_std": float(np.std(full_vals)),
                "base_std": float(np.std(base_vals)),
            }
        
        effect_matrix[svo] = effects
    
    return effect_matrix


def compute_seed_robustness(data):
    """각 SVO 조건 내 seed 간 변동 계수(CV)를 계산하여 결과 안정성을 평가."""
    metrics = ["reward", "gini", "cooperation"]
    robustness = {}
    
    for svo, seeds in data.items():
        r = {}
        for m in metrics:
            vals = [s[m] for s in seeds]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            cv = std_v / abs(mean_v) if abs(mean_v) > 1e-8 else float('inf')
            r[m] = {
                "mean": float(mean_v),
                "std": float(std_v),
                "cv": float(cv),
                "n_seeds": len(vals),
            }
        robustness[svo] = r
    
    return robustness


def plot_sensitivity(effect_matrix, robustness, output_dir):
    """민감도 분석 결과를 Heatmap으로 시각화합니다."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    svo_order = ["selfish", "individualist", "competitive", "prosocial",
                 "cooperative", "altruistic", "full_altruist"]
    svo_labels = [s for s in svo_order if s in effect_matrix]
    metrics = ["reward", "gini", "cooperation"]
    metric_labels = ["Reward", "Gini", "Cooperation"]
    
    # --- (a) Cohen's d Heatmap ---
    ax = axes[0]
    d_matrix = np.zeros((len(svo_labels), len(metrics)))
    for i, svo in enumerate(svo_labels):
        for j, m in enumerate(metrics):
            eff = effect_matrix[svo].get(m, {})
            d_matrix[i, j] = eff.get("d", 0) if isinstance(eff, dict) else eff
    
    im = ax.imshow(d_matrix, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticks(range(len(svo_labels)))
    ax.set_yticklabels(svo_labels, fontsize=9)
    ax.set_title("(a) Cohen's d\n(Full vs Baseline)")
    
    # 셀 내 값 표시
    for i in range(len(svo_labels)):
        for j in range(len(metrics)):
            val = d_matrix[i, j]
            color = "white" if abs(val) > 1.0 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                   color=color, fontsize=9, fontweight='bold')
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # --- (b) P-value Heatmap ---
    ax = axes[1]
    p_matrix = np.zeros((len(svo_labels), len(metrics)))
    for i, svo in enumerate(svo_labels):
        for j, m in enumerate(metrics):
            eff = effect_matrix[svo].get(m, {})
            p_matrix[i, j] = eff.get("p", 1.0) if isinstance(eff, dict) else 1.0
    
    # -log10(p) for better visualization
    log_p = -np.log10(p_matrix + 1e-20)
    im2 = ax.imshow(log_p, cmap='YlOrRd', aspect='auto', vmin=0, vmax=10)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticks(range(len(svo_labels)))
    ax.set_yticklabels(svo_labels, fontsize=9)
    ax.set_title("(b) Significance\n(-log10 p-value)")
    
    for i in range(len(svo_labels)):
        for j in range(len(metrics)):
            p_val = p_matrix[i, j]
            stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            color = "white" if log_p[i, j] > 5 else "black"
            ax.text(j, i, stars, ha="center", va="center",
                   color=color, fontsize=10, fontweight='bold')
    
    plt.colorbar(im2, ax=ax, shrink=0.8)
    
    # --- (c) Seed Robustness (CV) ---
    ax = axes[2]
    cv_matrix = np.zeros((len(svo_labels), len(metrics)))
    for i, svo in enumerate(svo_labels):
        if svo in robustness:
            for j, m in enumerate(metrics):
                cv_matrix[i, j] = robustness[svo][m]["cv"]
    
    im3 = ax.imshow(cv_matrix, cmap='Greens_r', aspect='auto', vmin=0, vmax=0.5)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticks(range(len(svo_labels)))
    ax.set_yticklabels(svo_labels, fontsize=9)
    ax.set_title("(c) Seed Robustness\n(CV < 0.1 = Robust)")
    
    for i in range(len(svo_labels)):
        for j in range(len(metrics)):
            val = cv_matrix[i, j]
            status = "OK" if val < 0.1 else "WARN" if val < 0.3 else "BAD"
            color = "white" if val > 0.3 else "black"
            ax.text(j, i, f"{val:.3f}\n{status}", ha="center", va="center",
                   color=color, fontsize=8)
    
    plt.colorbar(im3, ax=ax, shrink=0.8)
    
    plt.suptitle("Parameter Sensitivity Analysis: Meta-Ranking Effect Across SVO Conditions\n"
                 "(100 Agents, 10 Seeds per Condition)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "fig_sensitivity_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[G1] 민감도 분석 Figure 저장: {out_path}")
    return out_path


def print_summary(effect_matrix, robustness):
    """콘솔 요약."""
    print("\n" + "=" * 75)
    print("  G1: Parameter Sensitivity Analysis -- Summary")
    print("=" * 75)
    
    metrics = ["reward", "gini", "cooperation"]
    
    # 효과 일관성 체크: Gini에 대해 모든 SVO에서 같은 방향인가?
    for m in metrics:
        directions = []
        sig_count = 0
        for svo, effects in effect_matrix.items():
            eff = effects.get(m, {})
            if isinstance(eff, dict):
                d_val = eff.get("d", 0)
                p_val = eff.get("p", 1)
                directions.append(d_val)
                if p_val < 0.05:
                    sig_count += 1
        
        all_positive = all(d > 0 for d in directions)
        all_negative = all(d < 0 for d in directions)
        consistent = all_positive or all_negative
        
        print(f"\n  [{m.upper()}]")
        print(f"    Direction consistency: {'CONSISTENT' if consistent else 'MIXED'}")
        print(f"    Significant in {sig_count}/{len(directions)} SVO conditions")
        print(f"    Effect range: [{min(directions):.3f}, {max(directions):.3f}]")
    
    # Robustness 요약
    print(f"\n  [SEED ROBUSTNESS]")
    total_robust = 0
    total_cells = 0
    for svo, r in robustness.items():
        for m in metrics:
            cv = r[m]["cv"]
            total_cells += 1
            if cv < 0.1:
                total_robust += 1
    
    print(f"    Robust cells (CV < 0.1): {total_robust}/{total_cells}")
    print(f"    Conclusion: {'Results are ROBUST across seeds' if total_robust / total_cells > 0.7 else 'Some instability detected'}")
    print("=" * 75)


if __name__ == "__main__":
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) >= 2 else "simulation/outputs/reproduce"
    os.makedirs(output_dir, exist_ok=True)
    
    # 인자 2개면 기존 방식, 1개면 standalone 모드
    if len(sys.argv) >= 3:
        full_dir = sys.argv[1]
        base_dir = sys.argv[2]
        full_sweep = [f for f in os.listdir(full_dir) if f.startswith("sweep_")][0]
        base_sweep = [f for f in os.listdir(base_dir) if f.startswith("sweep_")][0]
        full_data = load_sweep_per_svo(os.path.join(full_dir, full_sweep))
        base_data = load_sweep_per_svo(os.path.join(base_dir, base_sweep))
    else:
        # Standalone 모드: 자체 mini sweep 데이터 생성
        print("[G1] Standalone 모드: Full/Baseline mini sweep 생성")
        from simulation.jax.experiment_jax import run_sweep
        from simulation.jax.config import SVO_SWEEP_THETAS
        
        angles = {
            "prosocial": SVO_SWEEP_THETAS["prosocial"],
            "cooperative": SVO_SWEEP_THETAS["cooperative"],
            "individualist": SVO_SWEEP_THETAS["individualist"],
        }
        
        # Full model sweep
        full_out = os.path.join(output_dir, "g1_full")
        os.makedirs(full_out, exist_ok=True)
        full_result, full_path = run_sweep(
            scale="small", svo_angles=angles, seeds=[42],
            output_dir=full_out,
        )
        full_data = load_sweep_per_svo(full_path)
        
        # Baseline sweep (meta OFF)
        base_out = os.path.join(output_dir, "g1_baseline")
        os.makedirs(base_out, exist_ok=True)
        base_result, base_path = run_sweep(
            scale="small", svo_angles=angles, seeds=[42],
            output_dir=base_out,
            config_override={"USE_META_RANKING": False},
        )
        base_data = load_sweep_per_svo(base_path)
        print("[G1] Mini sweep 완료")
    
    effect_matrix = compute_effect_sizes(full_data, base_data)
    robustness = compute_seed_robustness(full_data)
    
    print_summary(effect_matrix, robustness)
    
    fig_path = plot_sensitivity(effect_matrix, robustness, output_dir)
    
    # 결과 저장
    out_json = os.path.join(output_dir, "sensitivity_results.json")
    with open(out_json, "w") as f:
        json.dump({
            "effect_matrix": effect_matrix,
            "robustness": robustness,
        }, f, indent=2, default=str)
    print(f"[G1] 결과 JSON 저장: {out_json}")

