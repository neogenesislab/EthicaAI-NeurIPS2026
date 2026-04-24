"""
EthicaAI Human-AI Behavioral Comparison
Post-NeurIPS Research Feature

Zenodo Public Goods Game Dataset (2025)과 EthicaAI 시뮬레이션 결과를 비교.
Wasserstein Distance를 사용하여 AI 행동 분포가 인간과 얼마나 유사한지 정량화.
"""
import os
import json
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Placeholder for Human Data Loading
def load_human_pgg_data(dataset_path):
    """
    Zenodo/OSF 등에서 다운로드한 인간 PGG 데이터를 로드.
    반환 형식: { "cooperation_rate": [list of floats], "gini": [list of floats] }
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Generating mock data for testing.")
        # Mock Data (based on typical PGG findings: initial coop ~50%, decaying to ~10-20%)
        return {
            "cooperation_rate": np.random.beta(2, 5, 100).tolist(),
            "gini": np.random.normal(0.1, 0.05, 100).tolist()
        }
    # TODO: Implement actual CSV/Excel loading logic
    return {}

def load_ai_results(sweep_path):
    """EthicaAI 실험 결과 로드"""
    # Assuming standard sweep output structure
    with open(sweep_path, 'r') as f:
        data = json.load(f)
    
    metrics = {"cooperation_rate": [], "gini": []}
    for svo in data:
        runs = data[svo]["runs"]
        for r in runs:
            metrics["cooperation_rate"].append(r["metrics"]["cooperation_rate"][-1])
            metrics["gini"].append(r["metrics"]["gini"][-1])
    return metrics

def calculate_similarity(human_dist, ai_dist):
    """
    Wasserstein Distance (Earth Mover's Distance) 계산.
    0에 가까울수록 두 분포가 유사함.
    """
    return stats.wasserstein_distance(human_dist, ai_dist)

def plot_distributions(human_data, ai_data, metric, output_dir):
    """인간 vs AI 분포 비교 플롯"""
    plt.figure(figsize=(8, 5))
    
    plt.hist(human_data, bins=20, alpha=0.5, label='Human (PGG Dataset)', density=True, color='grey')
    plt.hist(ai_data, bins=20, alpha=0.5, label='AI (Meta-Ranking)', density=True, color='#2196F3')
    
    wd = calculate_similarity(human_data, ai_data)
    plt.title(f"Human-AI Comparison: {metric}\nWasserstein Distance = {wd:.4f}")
    plt.xlabel(metric)
    plt.ylabel("Density")
    plt.legend()
    
    path = os.path.join(output_dir, f"human_ai_{metric}.png")
    plt.savefig(path)
    plt.close()
    return path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python human_ai_comparison.py <human_data_path> <ai_sweep_json>")
        sys.exit(1)
        
    human_path = sys.argv[1]
    ai_path = sys.argv[2]
    output_dir = os.path.dirname(ai_path)
    
    human_metrics = load_human_pgg_data(human_path)
    ai_metrics = load_ai_results(ai_path)
    
    print("Comparing Distributions...")
    for metric in ["cooperation_rate", "gini"]:
        if metric in human_metrics and metric in ai_metrics:
            wd = calculate_similarity(human_metrics[metric], ai_metrics[metric])
            print(f"[{metric}] Wasserstein Distance: {wd:.4f}")
            plot_distributions(human_metrics[metric], ai_metrics[metric], metric, output_dir)
            
    print("Comparison Complete.")
