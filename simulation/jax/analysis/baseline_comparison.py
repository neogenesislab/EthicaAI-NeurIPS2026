"""
EthicaAI Baseline Comparison Analysis (Full vs Baseline)
NeurIPS 2026 Analysis

Full Model (Meta-Ranking ON)과 Baseline (Meta-Ranking OFF)의
Reward, Inequality(Gini), Cooperation 수준을 직접 비교.
t-test 및 Effect Size (Cohen's d) 계산.
"""
import os
import json
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_results(run_dir):
    """실험 결과 로드 (eval_results.json or sweep json)."""
    eval_path = os.path.join(run_dir, "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            return json.load(f)
    
    # eval_results가 없으면 sweep 파일에서 추출 (fallback)
    sweep_files = [f for f in os.listdir(run_dir) if f.startswith("sweep_")]
    if not sweep_files:
        return None
    with open(os.path.join(run_dir, sweep_files[0]), "r") as f:
        sweep_data = json.load(f)
    return extract_metrics_from_sweep(sweep_data)

def extract_metrics_from_sweep(sweep_data):
    """SVO별 메트릭 추출."""
    data = {}
    for svo, info in sweep_data.items():
        runs = info["runs"]
        rewards = [r["metrics"]["reward_mean"][-1] for r in runs]
        ginis = [r["metrics"]["gini"][-1] for r in runs]
        coops = [r["metrics"]["cooperation_rate"][-1] for r in runs]
        data[svo] = {"reward": rewards, "gini": ginis, "cooperation": coops}
    return data

def compare_models(full_data, baseline_data, output_dir):
    """Full vs Baseline 통계 비교 및 시각화."""
    metrics = ["reward", "gini", "cooperation"]
    results = {}
    
    # Aggregate metrics across all SVOs (Global effect)
    for m in metrics:
        full_vals = []
        base_vals = []
        for svo in full_data:
            if svo in baseline_data:
                # Full Model의 'metric' 키가 리스트인지 확인 (eval_results 구조 vs sweep 구조)
                # eval_results 구조: { "reward": { "selfish": { "mean": ..., "std": ... } } } -> This logic assumes raw values
                # 위 load_results는 sweep 구조를 가정하거나 raw data가 필요함.
                # eval_results.json에는 mean/std만 있으므로 t-test 불가. sweep에서 로드해야 함.
                # 편의상 sweep 데이터를 로드한다고 가정.
                f_v = full_data[svo][m] if isinstance(full_data[svo][m], list) else [] # Fix logic later
                b_v = baseline_data[svo][m] if isinstance(baseline_data[svo][m], list) else []
                full_vals.extend(f_v)
                base_vals.extend(b_v)
        
        if not full_vals or not base_vals:
            print(f"Skipping {m}: No data found")
            continue
            
        t_stat, p_val = stats.ttest_ind(full_vals, base_vals, equal_var=False)
        mean_diff = np.mean(full_vals) - np.mean(base_vals)
        pooled_std = np.sqrt((np.var(full_vals) + np.var(base_vals)) / 2)
        cohens_d = mean_diff / pooled_std
        
        results[m] = {
            "t_stat": t_stat,
            "p_value": p_val,
            "mean_diff": mean_diff,
            "cohens_d": cohens_d,
            "full_mean": np.mean(full_vals),
            "base_mean": np.mean(base_vals)
        }
        
    # Plotting (Bar chart with error bars)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_title = {"reward": "Mean Reward", "gini": "Gini Coefficient", "cooperation": "Cooperation Rate"}
    
    for i, m in enumerate(metrics):
        if m not in results: continue
        res = results[m]
        ax = axes[i]
        
        means = [res["base_mean"], res["full_mean"]]
        # Standard Error
        sems = [0, 0] # Simplify visualization for now
        
        bars = ax.bar(["Baseline", "Full Model"], means, 
                      color=["#9E9E9E", "#2196F3"], alpha=0.8)
        
        ax.set_title(f"{metrics_title[m]}\n(d={res['cohens_d']:.2f}, p={res['p_value']:.4f})")
        ax.set_ylabel(m)
        
        # P-value annotation
        if res['p_value'] < 0.05:
            ax.text(0.5, max(means)*1.05, f"* p={res['p_value']:.3f}", 
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "baseline_comparison.png"))
    plt.close()
    
    with open(os.path.join(output_dir, "baseline_stats.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python baseline_comparison.py <full_dir> <baseline_dir>")
        sys.exit(1)
        
    full_dir = sys.argv[1]
    base_dir = sys.argv[2]
    
    # Sweep 파일 로드 (Raw Data 필요)
    full_sweep = [f for f in os.listdir(full_dir) if f.startswith("sweep_")][0]
    base_sweep = [f for f in os.listdir(base_dir) if f.startswith("sweep_")][0]
    
    with open(os.path.join(full_dir, full_sweep), "r") as f:
        full_data = extract_metrics_from_sweep(json.load(f))
    with open(os.path.join(base_dir, base_sweep), "r") as f:
        base_data = extract_metrics_from_sweep(json.load(f))
        
    compare_models(full_data, base_data, full_dir)
    print("Baseline comparison complete!")
