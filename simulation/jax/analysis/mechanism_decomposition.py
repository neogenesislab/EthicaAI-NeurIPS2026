"""
EthicaAI Phase H2: Mechanism Decomposition Analysis
NeurIPS 2026 — 후속 연구

Meta-Ranking의 효과를 3가지 독립 메커니즘으로 분해하여
각각의 기여도를 정량화합니다:

1. SVO Rotation (보상 변환): 타인의 보상을 고려
2. Dynamic Lambda (상황적 조정): 자원 상태에 따른 커밋먼트 조절
3. Self-Control Cost (자제 비용): 이기적 충동을 억제하는 비용

이를 통해 "무엇이 진짜 효과를 내는가?"를 정량적으로 설명합니다.
"""
import os
import json
import math
import itertools
import numpy as np
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
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# PGG 환경 재사용
PGG_PARAMS = {
    "N_AGENTS": 4,
    "ENDOWMENT": 20.0,
    "MULTIPLIER": 1.6,
    "NUM_ROUNDS": 10,
    "NUM_SEEDS": 20,
}

SVO_THETA = math.pi / 4  # Prosocial


def run_pgg_with_mechanisms(svo_rotation=True, dynamic_lambda=True,
                            self_control=True, seed=0):
    """
    선택적으로 메커니즘을 켜/끄며 PGG를 시뮬레이션합니다.
    
    Args:
        svo_rotation: True면 SVO θ 적용, False면 θ=0 (selfish)
        dynamic_lambda: True면 wealth-dependent λ, False면 static λ
        self_control: True면 ψ cost 적용, False면 ψ=0
    """
    rng = np.random.RandomState(seed)
    n = PGG_PARAMS["N_AGENTS"]
    endowment = PGG_PARAMS["ENDOWMENT"]
    multiplier = PGG_PARAMS["MULTIPLIER"]
    num_rounds = PGG_PARAMS["NUM_ROUNDS"]
    beta = 0.1 if self_control else 0.0
    theta = SVO_THETA if svo_rotation else 0.0
    
    wealth = np.zeros(n)
    total_payoffs = np.zeros(n)
    contribution_rates = []
    
    for t in range(num_rounds):
        contributions = []
        for i in range(n):
            lam_base = math.sin(theta)
            
            if dynamic_lambda:
                if wealth[i] < -5.0:
                    lam = 0.0
                elif wealth[i] > 5.0:
                    lam = min(1.0, 1.5 * lam_base)
                else:
                    lam = lam_base
            else:
                lam = lam_base
            
            # Self-control cost에 의한 기여 조정
            # 높은 self-control cost → 기여 억제
            cost_factor = 1.0 / (1.0 + beta * abs(lam))
            adjusted_lam = lam * cost_factor
            
            noise = rng.normal(0, endowment * 0.03)
            c = np.clip(adjusted_lam * endowment + noise, 0, endowment)
            contributions.append(c)
        
        contributions = np.array(contributions)
        total_contrib = np.sum(contributions)
        public_good = total_contrib * multiplier / n
        payoffs = (endowment - contributions) + public_good
        
        wealth += payoffs - endowment
        total_payoffs += payoffs
        contribution_rates.append(float(np.mean(contributions) / endowment))
    
    return {
        "avg_contribution": float(np.mean(contribution_rates)),
        "avg_payoff": float(np.mean(total_payoffs) / num_rounds),
        "final_gini": float(compute_gini(total_payoffs)),
        "contribution_curve": contribution_rates,
        "welfare": float(np.sum(total_payoffs)),
    }


def compute_gini(values):
    """Gini coefficient."""
    values = np.sort(values)
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * values) / (n * np.sum(values)))


def run_decomposition(output_dir):
    """모든 메커니즘 조합을 테스트하는 완전 요인 설계(Full Factorial)."""
    mechanisms = ["svo_rotation", "dynamic_lambda", "self_control"]
    combinations = list(itertools.product([False, True], repeat=3))
    
    results = {}
    
    print("=" * 70)
    print("  H2: Mechanism Decomposition (Full Factorial Design)")
    print("=" * 70)
    print(f"  {'Config':30s} | Contrib | Payoff | Gini")
    print("-" * 70)
    
    for combo in combinations:
        svo, dyn, sc = combo
        label = f"SVO={'ON' if svo else 'OFF'}_Dyn={'ON' if dyn else 'OFF'}_SC={'ON' if sc else 'OFF'}"
        
        seed_results = []
        for seed in range(PGG_PARAMS["NUM_SEEDS"]):
            r = run_pgg_with_mechanisms(svo, dyn, sc, seed)
            seed_results.append(r)
        
        agg = {
            "svo_rotation": svo,
            "dynamic_lambda": dyn,
            "self_control": sc,
            "avg_contribution": float(np.mean([r["avg_contribution"] for r in seed_results])),
            "avg_payoff": float(np.mean([r["avg_payoff"] for r in seed_results])),
            "gini": float(np.mean([r["final_gini"] for r in seed_results])),
            "welfare": float(np.mean([r["welfare"] for r in seed_results])),
        }
        results[label] = agg
        
        print(f"  {label:30s} | {agg['avg_contribution']:.4f} | "
              f"{agg['avg_payoff']:.4f} | {agg['gini']:.4f}")
    
    # 각 메커니즘의 주효과(Main Effect) 계산
    print("\n--- Main Effects (Average Marginal Contribution) ---")
    for i, mech in enumerate(mechanisms):
        on_values = [v["avg_contribution"] for k, v in results.items()
                    if list(itertools.product([False, True], repeat=3))[
                        list(results.keys()).index(k)][i]]
        off_values = [v["avg_contribution"] for k, v in results.items()
                     if not list(itertools.product([False, True], repeat=3))[
                         list(results.keys()).index(k)][i]]
        
        effect = np.mean(on_values) - np.mean(off_values)
        print(f"  {mech:20s}: {'+' if effect > 0 else ''}{effect:.4f}")
    
    return results


def plot_decomposition(results, output_dir):
    """메커니즘 분해 시각화."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    labels = list(results.keys())
    short_labels = []
    for label in labels:
        parts = label.split("_")
        short = ""
        for i in range(0, len(parts), 2):
            key = parts[i][:3]
            val = parts[i+1] if i+1 < len(parts) else ""
            short += f"{key}{'✓' if val == 'ON' else '✗'}"
        short_labels.append(label.replace("_", "\n"))
    
    metrics = {
        "avg_contribution": "Contribution Rate",
        "avg_payoff": "Average Payoff",
        "gini": "Gini Coefficient",
    }
    
    for i, (metric_key, metric_name) in enumerate(metrics.items()):
        ax = axes[i]
        values = [results[l][metric_key] for l in labels]
        
        # 색상: 메커니즘 조합에 따른 그라데이션
        n_on = [sum([results[l]["svo_rotation"], results[l]["dynamic_lambda"],
                      results[l]["self_control"]]) for l in labels]
        colors_map = {0: '#F44336', 1: '#FF9800', 2: '#4CAF50', 3: '#2196F3'}
        bar_colors = [colors_map[n] for n in n_on]
        
        bars = ax.bar(range(len(labels)), values, color=bar_colors, alpha=0.8,
                     edgecolor='white', linewidth=0.5)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([f"{'S' if results[l]['svo_rotation'] else '-'}"
                           f"{'D' if results[l]['dynamic_lambda'] else '-'}"
                           f"{'C' if results[l]['self_control'] else '-'}"
                           for l in labels], fontsize=9)
        ax.set_ylabel(metric_name)
        ax.set_title(f"({chr(97+i)}) {metric_name}")
        ax.grid(True, alpha=0.3)
        
        # 값 표시
        for j, v in enumerate(values):
            ax.text(j, v, f"{v:.3f}", ha='center', va='bottom', fontsize=7)
    
    # 범례
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#F44336', label='0 mechanisms'),
        Patch(facecolor='#FF9800', label='1 mechanism'),
        Patch(facecolor='#4CAF50', label='2 mechanisms'),
        Patch(facecolor='#2196F3', label='All 3 mechanisms'),
    ]
    axes[2].legend(handles=legend_elements, fontsize=8, loc='upper right')
    
    plt.suptitle("H2: Mechanism Decomposition\n"
                 "S=SVO Rotation, D=Dynamic Lambda, C=Self-Control Cost",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "fig_mechanism_decomposition.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[H2] Figure 저장: {out_path}")
    return out_path


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "simulation/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    results = run_decomposition(output_dir)
    plot_decomposition(results, output_dir)
    
    out_json = os.path.join(output_dir, "mechanism_decomposition_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[H2] 결과 JSON 저장: {out_json}")
