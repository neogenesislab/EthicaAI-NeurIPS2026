"""
Q5: Mechanistic Interpretability — λ_t 결정 회로 시각화
EthicaAI Phase Q — 해석 가능성

메타랭킹 에이전트의 λ_t 결정 과정을 분해하고 시각화:
1. Feature Attribution: 어떤 입력이 λ_t에 가장 영향력 있는가
2. Phase Space: (resource, λ_t) 위상 공간 궤적
3. Decision Boundary: 자원-SVO 결정 경계

출력: Fig 65 (기여도 분석), Fig 66 (위상 공간 + 결정 경계)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json, os, sys

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "ETHICAAI_OUTPUT_DIR", "simulation/outputs/reproduce")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENDOWMENT = 100.0
N_STEPS = 200


def lambda_function(svo_deg, resource, prev_lambda, neighbor_avg=None):
    """메타랭킹 λ_t 계산 (분해 가능한 형태)"""
    base = np.sin(np.radians(svo_deg))
    
    # 자원 반응 (resource_modifier)
    if resource < 0.2:
        resource_mod = 0.3
    elif resource > 0.7:
        resource_mod = 1.5
    else:
        resource_mod = 0.7 + 1.6 * resource
    
    # 이웃 영향 (social_influence)
    if neighbor_avg is not None:
        social_mod = 0.5 + 0.5 * neighbor_avg
    else:
        social_mod = 1.0
    
    target = np.clip(base * resource_mod * social_mod, 0, 1)
    alpha = 0.1
    new_lambda = (1 - alpha) * prev_lambda + alpha * target
    
    return new_lambda, {
        "base_svo": float(base),
        "resource_mod": float(resource_mod),
        "social_mod": float(social_mod),
        "target": float(target),
        "momentum_effect": float((1 - alpha) * prev_lambda),
        "update_effect": float(alpha * target),
    }


def feature_attribution():
    """입력 요소별 λ_t 기여도 분석"""
    svo_range = np.linspace(0, 90, 20)
    resource_range = np.linspace(0, 1, 20)
    
    attributions = {"svo": [], "resource": [], "momentum": [], "social": []}
    
    for svo in svo_range:
        for resource in resource_range:
            _, decomp = lambda_function(svo, resource, 0.5, 0.5)
            total = abs(decomp["base_svo"]) + abs(decomp["resource_mod"] - 1) + \
                    abs(decomp["momentum_effect"] - 0.45) + abs(decomp["social_mod"] - 1)
            
            if total > 0:
                attributions["svo"].append(abs(decomp["base_svo"]) / total)
                attributions["resource"].append(abs(decomp["resource_mod"] - 1) / total)
                attributions["momentum"].append(abs(decomp["momentum_effect"] - 0.45) / total)
                attributions["social"].append(abs(decomp["social_mod"] - 1) / total)
    
    return {k: float(np.mean(v)) for k, v in attributions.items()}


def phase_space_trajectories():
    """(resource, λ_t) 위상 공간 궤적"""
    trajectories = {}
    regimes = {
        "crisis": {"init_resource": 0.1, "svo": 45.0},
        "abundance": {"init_resource": 0.9, "svo": 45.0},
        "normal": {"init_resource": 0.5, "svo": 45.0},
        "selfish_crisis": {"init_resource": 0.1, "svo": 0.0},
        "altruist_normal": {"init_resource": 0.5, "svo": 90.0},
    }
    
    for name, config in regimes.items():
        resource = config["init_resource"]
        lambda_t = np.sin(np.radians(config["svo"]))
        
        r_hist, l_hist = [resource], [lambda_t]
        
        for t in range(N_STEPS):
            lambda_t, _ = lambda_function(config["svo"], resource, lambda_t)
            # 자원 업데이트
            contrib_rate = lambda_t * 0.8
            resource = np.clip(resource + 0.02 * (contrib_rate - 0.3), 0, 1)
            r_hist.append(float(resource))
            l_hist.append(float(lambda_t))
        
        trajectories[name] = {"resource": r_hist, "lambda": l_hist}
    
    return trajectories


def decision_boundary():
    """SVO-Resource 결정 경계 맵"""
    svo_range = np.linspace(0, 90, 100)
    resource_range = np.linspace(0, 1, 100)
    
    lambda_map = np.zeros((100, 100))
    contrib_map = np.zeros((100, 100))
    
    for i, svo in enumerate(svo_range):
        for j, resource in enumerate(resource_range):
            lambda_t, _ = lambda_function(svo, resource, np.sin(np.radians(svo)))
            lambda_map[j, i] = lambda_t
            contrib_map[j, i] = lambda_t * ENDOWMENT * 0.8
    
    return svo_range, resource_range, lambda_map, contrib_map


def plot_fig65(attributions, trajectories):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Fig 65: Mechanistic Interpretability — λ_t Decision Circuit",
                 fontsize=14, fontweight='bold', y=1.02)
    
    # 좌: Feature Attribution 파이
    ax = axes[0]
    labels = ['SVO (θ)', 'Resource (R)', 'Momentum (λ_{t-1})', 'Social (neighbors)']
    values = [attributions["svo"], attributions["resource"], attributions["momentum"], attributions["social"]]
    colors = ['#1e88e5', '#43a047', '#ff9800', '#7e57c2']
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontsize': 9})
    ax.set_title('Feature Attribution to λ_t', fontweight='bold')
    
    # 중: 위상 공간 궤적
    ax = axes[1]
    traj_colors = {'crisis': '#e53935', 'abundance': '#1e88e5', 'normal': '#43a047',
                   'selfish_crisis': '#ff9800', 'altruist_normal': '#7e57c2'}
    for name, traj in trajectories.items():
        ax.plot(traj["resource"], traj["lambda"], linewidth=1.5, alpha=0.7,
               color=traj_colors[name], label=name.replace('_', ' ').title())
        ax.plot(traj["resource"][0], traj["lambda"][0], 'o', color=traj_colors[name], markersize=8)
        ax.plot(traj["resource"][-1], traj["lambda"][-1], 's', color=traj_colors[name], markersize=8)
    ax.set_title('Phase Space: (Resource, λ_t) Trajectories', fontweight='bold')
    ax.set_xlabel('Resource Level'); ax.set_ylabel('λ_t')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # 우: 구성 요소 분해 시계열
    ax = axes[2]
    resource = 0.1
    lambda_t = 0.5
    base_hist, res_hist, mom_hist, upd_hist = [], [], [], []
    
    for t in range(100):
        lambda_t, decomp = lambda_function(45.0, resource, lambda_t)
        base_hist.append(decomp["base_svo"])
        res_hist.append(decomp["resource_mod"])
        mom_hist.append(decomp["momentum_effect"])
        upd_hist.append(decomp["update_effect"])
        resource = np.clip(resource + 0.02 * (lambda_t * 0.8 - 0.3), 0, 1)
    
    ax.stackplot(range(100), mom_hist, upd_hist, labels=['Momentum', 'Update'],
                colors=['#ff9800', '#1e88e5'], alpha=0.7)
    ax.plot(range(100), [m + u for m, u in zip(mom_hist, upd_hist)], 'k-', linewidth=2, label='λ_t')
    ax.set_title('Component Decomposition (Crisis → Recovery)', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Value')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig65_interpretability.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[Q5] Fig 65 저장: {path}")


def plot_fig66(svo_range, resource_range, lambda_map, contrib_map):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Fig 66: Decision Boundary — SVO × Resource → λ_t & Contribution",
                 fontsize=14, fontweight='bold', y=1.02)
    
    ax = axes[0]
    im = ax.imshow(lambda_map, origin='lower', cmap='viridis', aspect='auto',
                   extent=[0, 90, 0, 1])
    ax.contour(svo_range, resource_range, lambda_map, levels=[0.1, 0.3, 0.5, 0.7], colors='white', linewidths=1.5)
    ax.set_title('λ_t Decision Map', fontweight='bold')
    ax.set_xlabel('SVO θ (degrees)'); ax.set_ylabel('Resource Level')
    plt.colorbar(im, ax=ax, label='λ_t')
    
    ax = axes[1]
    im = ax.imshow(contrib_map, origin='lower', cmap='YlOrRd', aspect='auto',
                   extent=[0, 90, 0, 1])
    ax.contour(svo_range, resource_range, contrib_map, levels=[10, 30, 50, 70], colors='black', linewidths=1.5)
    ax.set_title('Contribution Decision Map', fontweight='bold')
    ax.set_xlabel('SVO θ (degrees)'); ax.set_ylabel('Resource Level')
    plt.colorbar(im, ax=ax, label='Contribution')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig66_decision_boundary.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[Q5] Fig 66 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [Q5] Mechanistic Interpretability")
    print("=" * 60)
    
    print("\n[1/3] Feature Attribution...")
    attr = feature_attribution()
    for k, v in attr.items():
        print(f"  {k:>12}: {v:.1%}")
    
    print("\n[2/3] Phase Space Trajectories...")
    traj = phase_space_trajectories()
    
    print("\n[3/3] Decision Boundary...")
    svo_r, res_r, lmap, cmap = decision_boundary()
    
    plot_fig65(attr, traj)
    plot_fig66(svo_r, res_r, lmap, cmap)
    
    json_path = os.path.join(OUTPUT_DIR, "interpretability_results.json")
    with open(json_path, 'w') as f:
        json.dump({"attributions": attr}, f, indent=2)
    print(f"\n[Q5] JSON: {json_path}")
