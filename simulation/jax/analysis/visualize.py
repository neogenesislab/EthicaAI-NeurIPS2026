"""
EthicaAI Visualization Module
Generates publication-quality figures for the paper.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from typing import Dict, Any, List

# Style Configuration
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color Palette
COLORS = {
    "selfish": "#E74C3C",
    "competitive": "#E67E22",
    "individualist": "#F1C40F",
    "prosocial": "#2ECC71",
    "altruistic": "#3498DB",
    "full_altruist": "#9B59B6",
}

def _get_color(name):
    return COLORS.get(name, "#95A5A6")


def plot_learning_curves(sweep_results: Dict[str, Any], 
                          output_dir: str, metric_key="reward_mean"):
    """
    Fig.1: SVO 조건별 학습 곡선 (mean ± std).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, data in sweep_results.items():
        runs = data["runs"]
        # Stack metric across seeds: (S, T) where S=seeds, T=epochs
        metric_matrix = np.array([r["metrics"][metric_key] for r in runs])
        
        mean_curve = np.mean(metric_matrix, axis=0)
        std_curve = np.std(metric_matrix, axis=0)
        epochs = np.arange(len(mean_curve))
        
        color = _get_color(name)
        ax.plot(epochs, mean_curve, label=f"{name} (θ={data['theta']:.2f})", 
                color=color, linewidth=2)
        ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve,
                         alpha=0.15, color=color)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Learning Curves by SVO Condition")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(output_dir, "fig1_learning_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_cooperation_rate(sweep_results: Dict[str, Any], output_dir: str):
    """
    Fig.2: SVO 조건별 협력률 진화.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, data in sweep_results.items():
        runs = data["runs"]
        coop_matrix = np.array([r["metrics"]["cooperation_rate"] for r in runs])
        
        mean_curve = np.mean(coop_matrix, axis=0)
        std_curve = np.std(coop_matrix, axis=0)
        epochs = np.arange(len(mean_curve))
        
        color = _get_color(name)
        ax.plot(epochs, mean_curve, label=name, color=color, linewidth=2)
        ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve,
                         alpha=0.15, color=color)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cooperation Rate")
    ax.set_title("Cooperation Rate Evolution by SVO")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    path = os.path.join(output_dir, "fig2_cooperation_rate.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_threshold_evolution(sweep_results: Dict[str, Any], output_dir: str):
    """
    Fig.3: 역치 분화 과정 (Clean vs Harvest threshold).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, data in sweep_results.items():
        runs = data["runs"]
        clean_matrix = np.array([r["metrics"]["threshold_clean_mean"] for r in runs])
        harvest_matrix = np.array([r["metrics"]["threshold_harvest_mean"] for r in runs])
        
        color = _get_color(name)
        epochs = np.arange(clean_matrix.shape[1])
        
        # Clean threshold
        mean_c = np.mean(clean_matrix, axis=0)
        axes[0].plot(epochs, mean_c, label=name, color=color, linewidth=2)
        
        # Harvest threshold
        mean_h = np.mean(harvest_matrix, axis=0)
        axes[1].plot(epochs, mean_h, label=name, color=color, linewidth=2)
    
    axes[0].set_title("Clean Threshold (θ_clean)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean Threshold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("Harvest Threshold (θ_harvest)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Threshold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle("Adaptive Threshold Evolution (Division of Labor)", fontsize=14, y=1.02)
    
    path = os.path.join(output_dir, "fig3_threshold_evolution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_gini_comparison(sweep_results: Dict[str, Any], output_dir: str):
    """
    Fig.5: SVO 조건별 Gini 계수 비교 (Bar Plot).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    
    names = []
    means = []
    stds = []
    colors = []
    
    for name, data in sweep_results.items():
        runs = data["runs"]
        ginis = [r["metrics"]["gini"][-1] for r in runs]
        names.append(name)
        means.append(np.mean(ginis))
        stds.append(np.std(ginis))
        colors.append(_get_color(name))
    
    bars = ax.bar(names, means, yerr=stds, color=colors, 
                   capsize=5, alpha=0.85, edgecolor="white", linewidth=1.5)
    
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Reward Inequality by SVO Condition")
    ax.grid(True, alpha=0.3, axis='y')
    
    path = os.path.join(output_dir, "fig5_gini_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_svo_vs_welfare(sweep_results: Dict[str, Any], output_dir: str):
    """
    Fig.7: SVO vs 사회적 복지 (Scatter + Regression Line).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    thetas = []
    rewards = []
    
    for name, data in sweep_results.items():
        theta = data["theta"]
        for run in data["runs"]:
            thetas.append(theta)
            rewards.append(run["metrics"]["reward_mean"][-1])
    
    thetas = np.array(thetas)
    rewards = np.array(rewards)
    
    # Scatter
    ax.scatter(thetas, rewards, c='#2E86AB', s=80, alpha=0.7, edgecolors='white')
    
    # Regression Line
    if len(thetas) > 2:
        coeffs = np.polyfit(thetas, rewards, 2)
        x_fit = np.linspace(thetas.min(), thetas.max(), 100)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, '--', color='#E74C3C', linewidth=2, label="Quadratic Fit")
    
    ax.set_xlabel("SVO Angle (θ, radians)")
    ax.set_ylabel("Mean Reward")
    ax.set_title("SVO vs Social Welfare")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(output_dir, "fig7_svo_vs_welfare.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def generate_all_figures(sweep_results: Dict[str, Any], output_dir: str):
    """
    모든 논문 Figure 일괄 생성.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating figures to: {output_dir}")
    
    paths = {}
    paths["fig1"] = plot_learning_curves(sweep_results, output_dir)
    paths["fig2"] = plot_cooperation_rate(sweep_results, output_dir)
    paths["fig3"] = plot_threshold_evolution(sweep_results, output_dir)
    paths["fig5"] = plot_gini_comparison(sweep_results, output_dir)
    paths["fig7"] = plot_svo_vs_welfare(sweep_results, output_dir)
    
    print(f"\nTotal figures generated: {len(paths)}")
    return paths
