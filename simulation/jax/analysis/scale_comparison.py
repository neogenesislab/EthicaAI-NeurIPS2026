"""
EthicaAI Scale Comparison Analysis (20-Agent vs 100-Agent)
NeurIPS 2026 보강 Figure: Fig 10 (Scalability Analysis)

20-에이전트(Medium)와 100-에이전트(Large) 결과를 비교하여
메타랭킹 효과의 규모 독립성(Scale Invariance)을 검증.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Publication-ready Matplotlib 설정
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
})

# 색상 팔레트 (NeurIPS 스타일)
COLORS = {
    "medium": "#2196F3",    # 파랑 (20-agent)
    "large": "#FF5722",     # 주황 (100-agent)
    "baseline": "#9E9E9E",  # 회색 (baseline)
}


def load_results(run_dir):
    """실험 결과 로드."""
    sweep_files = [f for f in os.listdir(run_dir)
                   if f.startswith("sweep_") and f.endswith(".json")]
    if not sweep_files:
        return None
    with open(os.path.join(run_dir, sweep_files[0]), "r") as f:
        return json.load(f)


def extract_metrics(sweep_results):
    """SVO별 평균 메트릭 추출."""
    data = {}
    for svo_name, svo_data in sweep_results.items():
        theta = svo_data["theta"]
        runs = svo_data["runs"]
        
        rewards = [r["metrics"]["reward_mean"][-1] for r in runs if "metrics" in r]
        coops = [r["metrics"]["cooperation_rate"][-1] for r in runs if "metrics" in r]
        ginis = [r["metrics"]["gini"][-1] for r in runs if "metrics" in r]
        
        data[svo_name] = {
            "theta": theta,
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "coop_mean": np.mean(coops),
            "coop_std": np.std(coops),
            "gini_mean": np.mean(ginis),
            "gini_std": np.std(ginis),
            "n_runs": len(rewards),
        }
    return data


def plot_scale_comparison(medium_data, large_data, output_dir):
    """
    Fig 10: 20-Agent vs 100-Agent 비교 (3-panel)
    (a) Reward by SVO
    (b) Cooperation Rate by SVO
    (c) Gini by SVO
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # SVO 순서 고정
    svo_order = ["selfish", "individualist", "competitive",
                 "prosocial", "cooperative", "altruistic", "full_altruist"]
    svo_labels = ["SEL", "IND", "COM", "PRO", "COO", "ALT", "F-A"]
    
    metrics = [
        ("reward_mean", "reward_std", "Mean Reward", "(a) Reward by SVO"),
        ("coop_mean", "coop_std", "Cooperation Rate", "(b) Cooperation Rate by SVO"),
        ("gini_mean", "gini_std", "Gini Coefficient", "(c) Gini by SVO"),
    ]
    
    x = np.arange(len(svo_order))
    width = 0.35
    
    for ax, (mean_key, std_key, ylabel, title) in zip(axes, metrics):
        med_vals = [medium_data.get(s, {}).get(mean_key, 0) for s in svo_order]
        med_errs = [medium_data.get(s, {}).get(std_key, 0) for s in svo_order]
        lrg_vals = [large_data.get(s, {}).get(mean_key, 0) for s in svo_order]
        lrg_errs = [large_data.get(s, {}).get(std_key, 0) for s in svo_order]
        
        bars1 = ax.bar(x - width/2, med_vals, width, yerr=med_errs,
                       label="20-Agent", color=COLORS["medium"],
                       alpha=0.85, capsize=3, error_kw={"linewidth": 0.8})
        bars2 = ax.bar(x + width/2, lrg_vals, width, yerr=lrg_errs,
                       label="100-Agent", color=COLORS["large"],
                       alpha=0.85, capsize=3, error_kw={"linewidth": 0.8})
        
        ax.set_xlabel("SVO Condition")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(svo_labels, fontsize=9)
        ax.legend(fontsize=8, loc="best")
    
    plt.tight_layout()
    
    path = os.path.join(output_dir, "fig10_scale_comparison.pdf")
    plt.savefig(path, format="pdf")
    path_png = os.path.join(output_dir, "fig10_scale_comparison.png")
    plt.savefig(path_png, format="png", dpi=300)
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_ate_scale_comparison(medium_causal, large_causal, output_dir):
    """
    Fig 11: ATE 비교 (Forest Plot - 20 vs 100 Agent)
    같은 가설(H1, H2, H3)에 대해 두 스케일의 ATE를 나란히 비교.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    hypotheses = ["H1_reward", "H2_cooperation", "H3_gini"]
    labels = ["H1: SVO→Reward", "H2: SVO→Cooperation", "H3: SVO→Gini"]
    
    y_pos = np.arange(len(hypotheses))
    
    for i, (h_key, label) in enumerate(zip(hypotheses, labels)):
        # Medium (20-agent)
        med = medium_causal.get(h_key, {})
        med_ate = med.get("ate", 0)
        med_se = med.get("se", 0)
        med_sig = med.get("significant", False)
        
        # Large (100-agent)
        lrg = large_causal.get(h_key, {})
        lrg_ate = lrg.get("ate", 0)
        lrg_se = lrg.get("se", 0)
        lrg_sig = lrg.get("significant", False)
        
        # Plot
        offset = 0.15
        ax.errorbar(med_ate, i - offset, xerr=1.96*med_se, fmt="o",
                    color=COLORS["medium"], markersize=8, capsize=5,
                    label="20-Agent" if i == 0 else "")
        ax.errorbar(lrg_ate, i + offset, xerr=1.96*lrg_se, fmt="s",
                    color=COLORS["large"], markersize=8, capsize=5,
                    label="100-Agent" if i == 0 else "")
        
        # 유의성 표시
        for ate, y, sig, color in [(med_ate, i-offset, med_sig, COLORS["medium"]),
                                    (lrg_ate, i+offset, lrg_sig, COLORS["large"])]:
            if sig:
                ax.annotate("*", (ate, y), textcoords="offset points",
                           xytext=(0, 8), ha="center", fontsize=14,
                           fontweight="bold", color=color)
    
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Average Treatment Effect (ATE) with 95% CI")
    ax.set_title("Scale Comparison: ATE Consistency (20 vs 100 Agents)",
                fontweight="bold")
    ax.legend(loc="upper right")
    ax.invert_yaxis()
    
    plt.tight_layout()
    path = os.path.join(output_dir, "fig11_ate_scale_comparison.pdf")
    plt.savefig(path, format="pdf")
    path_png = os.path.join(output_dir, "fig11_ate_scale_comparison.png")
    plt.savefig(path_png, format="png", dpi=300)
    plt.close()
    print(f"Saved: {path}")
    return path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python scale_comparison.py <medium_run_dir> <large_run_dir>")
        print("Example: python scale_comparison.py simulation/outputs/run_medium_xxx simulation/outputs/run_large_xxx")
        sys.exit(1)
    
    medium_dir = sys.argv[1]
    large_dir = sys.argv[2]
    
    medium_results = load_results(medium_dir)
    large_results = load_results(large_dir)
    
    if not medium_results or not large_results:
        print("Could not load results!")
        sys.exit(1)
    
    medium_data = extract_metrics(medium_results)
    large_data = extract_metrics(large_results)
    
    output_dir = os.path.join(large_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_scale_comparison(medium_data, large_data, output_dir)
    
    # Causal 결과가 있으면 ATE 비교도
    med_causal_path = os.path.join(medium_dir, "causal_results.json")
    lrg_causal_path = os.path.join(large_dir, "causal_results.json")
    if os.path.exists(med_causal_path) and os.path.exists(lrg_causal_path):
        with open(med_causal_path, "r") as f:
            med_causal = json.load(f)
        with open(lrg_causal_path, "r") as f:
            lrg_causal = json.load(f)
        plot_ate_scale_comparison(med_causal, lrg_causal, output_dir)
    
    print("Scale comparison complete!")
