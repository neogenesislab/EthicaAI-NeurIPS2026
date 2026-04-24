"""
O3: AI 거버넌스 투표 게임 (AI Governance Voting Game)
EthicaAI Phase O — 연구축 II: 현실 사회 딜레마

N개 이해관계자가 AI 규제 수준을 투표로 결정하는 합의 게임.
메타랭킹(λ)이 합의 도달 속도와 규제 수준에 미치는 효과를 분석합니다.

출력: Fig 43 (합의 과정), Fig 44 (ATE)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import os
import sys

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "ETHICAAI_OUTPUT_DIR", "simulation/outputs/reproduce"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 이해관계자 설정 (하드코딩 금지) ===
STAKEHOLDER_CONFIGS = {
    "tech_corp":   {"power": 0.30, "risk_tol": 0.8, "ideal_reg": 2, "name": "Tech Corp"},
    "regulator":   {"power": 0.25, "risk_tol": 0.2, "ideal_reg": 7, "name": "Regulator"},
    "academia":    {"power": 0.15, "risk_tol": 0.5, "ideal_reg": 5, "name": "Academia"},
    "civil_soc":   {"power": 0.15, "risk_tol": 0.3, "ideal_reg": 6, "name": "Civil Society"},
    "startup":     {"power": 0.10, "risk_tol": 0.9, "ideal_reg": 1, "name": "AI Startup"},
    "labor_union": {"power": 0.05, "risk_tol": 0.1, "ideal_reg": 8, "name": "Labor Union"},
}

GOVERNANCE_PARAMS = {
    "max_regulation": 9,             # 0(무규제)~9(완전규제)
    "consensus_threshold": 0.65,     # 가중 동의 65%
    "innovation_cost_per_level": 0.05,
    "safety_gain_per_level": 0.08,
    "deadlock_penalty": -0.15,
    "compromise_speed": 0.1,         # 라운드당 타협 속도
}

SVO_CONDITIONS = {"selfish": 0.0, "individualist": 15.0, "prosocial": 45.0, "altruistic": 90.0}
N_ROUNDS = 100
N_SEEDS = 10


def simulate_governance(svo_theta_deg, seed, use_meta=True):
    """AI 거버넌스 투표 시뮬레이션"""
    rng = np.random.RandomState(seed)
    svo_theta = np.radians(svo_theta_deg)
    lambda_base = np.sin(svo_theta)
    
    stakeholders = list(STAKEHOLDER_CONFIGS.keys())
    n_sh = len(stakeholders)
    powers = np.array([STAKEHOLDER_CONFIGS[s]["power"] for s in stakeholders])
    ideals = np.array([STAKEHOLDER_CONFIGS[s]["ideal_reg"] for s in stakeholders], dtype=float)
    risk_tols = np.array([STAKEHOLDER_CONFIGS[s]["risk_tol"] for s in stakeholders])
    
    max_reg = GOVERNANCE_PARAMS["max_regulation"]
    consensus_thresh = GOVERNANCE_PARAMS["consensus_threshold"]
    
    # 현재 입장 (초기 = ideal)
    positions = ideals.copy()
    
    # 기록
    position_history = np.zeros((N_ROUNDS, n_sh))
    consensus_level = np.zeros(N_ROUNDS)
    welfare_history = np.zeros(N_ROUNDS)
    lambda_history = np.zeros((N_ROUNDS, n_sh))
    consensus_reached_at = N_ROUNDS  # 미도달 시
    
    for t in range(N_ROUNDS):
        # λ 결정
        lambdas = np.zeros(n_sh)
        for i in range(n_sh):
            if use_meta:
                # 교착 상태 감지 → λ 증가
                if t > 5:
                    position_variance = np.std(positions)
                    deadlock_risk = position_variance / (max_reg + 1e-8)
                    
                    if deadlock_risk > 0.3:
                        lambdas[i] = min(1.0, lambda_base * 1.5)  # 타협 가속
                    elif deadlock_risk < 0.1:
                        lambdas[i] = lambda_base  # 이미 수렴 중
                    else:
                        lambdas[i] = lambda_base
                else:
                    lambdas[i] = lambda_base
            else:
                lambdas[i] = lambda_base
        
        lambda_history[t] = lambdas
        
        # 투표 (가중 중위값)
        weighted_median = _weighted_median(positions, powers)
        
        # 타협 (각자 중위값 방향으로 이동)
        for i in range(n_sh):
            speed = GOVERNANCE_PARAMS["compromise_speed"] * (1 + lambdas[i])
            direction = weighted_median - positions[i]
            positions[i] += speed * direction + rng.normal(0, 0.1)
            positions[i] = np.clip(positions[i], 0, max_reg)
        
        position_history[t] = positions
        
        # 합의 판정
        agreement_weight = 0
        for i in range(n_sh):
            if abs(positions[i] - weighted_median) < 1.0:
                agreement_weight += powers[i]
        
        consensus_level[t] = agreement_weight
        
        if agreement_weight >= consensus_thresh and consensus_reached_at == N_ROUNDS:
            consensus_reached_at = t
        
        # 복지 계산
        reg_level = weighted_median
        innovation = 1.0 - GOVERNANCE_PARAMS["innovation_cost_per_level"] * reg_level
        safety = GOVERNANCE_PARAMS["safety_gain_per_level"] * reg_level
        welfare = innovation + safety
        if agreement_weight < consensus_thresh:
            welfare += GOVERNANCE_PARAMS["deadlock_penalty"]
        welfare_history[t] = welfare
    
    return {
        "positions": position_history,
        "consensus_level": consensus_level,
        "welfare": welfare_history,
        "lambdas": lambda_history,
        "consensus_at": int(consensus_reached_at),
        "final_regulation": float(np.mean(positions)),
        "final_welfare": float(np.mean(welfare_history[-20:])),
        "final_consensus": float(np.mean(consensus_level[-20:])),
    }


def _weighted_median(values, weights):
    """가중 중위값"""
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cum_weight = np.cumsum(sorted_weights)
    mid = np.sum(sorted_weights) / 2
    idx = np.searchsorted(cum_weight, mid)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def run_experiment():
    results = {}
    for svo_name, svo_theta in SVO_CONDITIONS.items():
        meta_runs = [simulate_governance(svo_theta, s, True) for s in range(N_SEEDS)]
        base_runs = [simulate_governance(svo_theta, s, False) for s in range(N_SEEDS)]
        results[svo_name] = {
            "meta_consensus_at": float(np.mean([r["consensus_at"] for r in meta_runs])),
            "base_consensus_at": float(np.mean([r["consensus_at"] for r in base_runs])),
            "meta_regulation": float(np.mean([r["final_regulation"] for r in meta_runs])),
            "base_regulation": float(np.mean([r["final_regulation"] for r in base_runs])),
            "meta_welfare": float(np.mean([r["final_welfare"] for r in meta_runs])),
            "base_welfare": float(np.mean([r["final_welfare"] for r in base_runs])),
            "ate_consensus_speed": float(np.mean([r["consensus_at"] for r in base_runs]) -
                                         np.mean([r["consensus_at"] for r in meta_runs])),  # 빨라지면 양수
            "ate_welfare": float(np.mean([r["final_welfare"] for r in meta_runs]) -
                                 np.mean([r["final_welfare"] for r in base_runs])),
            "meta_runs": meta_runs,
            "base_runs": base_runs,
        }
    return results


def plot_fig43(results):
    """Fig 43: AI 거버넌스 — 합의 과정"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fig 43: AI Governance Voting — Consensus Formation with Meta-Ranking",
                 fontsize=14, fontweight='bold', y=0.98)
    
    sh_names = [STAKEHOLDER_CONFIGS[s]["name"] for s in STAKEHOLDER_CONFIGS]
    sh_colors = plt.cm.Set2(np.linspace(0.1, 0.9, len(sh_names)))
    
    # 상좌: 이해관계자 입장 변화 (prosocial Meta)
    ax = axes[0, 0]
    meta_run = results["prosocial"]["meta_runs"][0]
    for i, name in enumerate(sh_names):
        ax.plot(meta_run["positions"][:, i], color=sh_colors[i], linewidth=2, label=name)
    ax.set_title('Stakeholder Positions (Prosocial Meta)', fontweight='bold')
    ax.set_xlabel('Round'); ax.set_ylabel('Regulation Level')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # 상우: 합의 수준 비교
    ax = axes[0, 1]
    colors_svo = {'selfish': '#e53935', 'individualist': '#ff9800', 'prosocial': '#1e88e5', 'altruistic': '#43a047'}
    for svo in SVO_CONDITIONS:
        meta_run = results[svo]["meta_runs"][0]
        ax.plot(meta_run["consensus_level"], color=colors_svo[svo], linewidth=2, label=f'{svo} (Meta)')
        base_run = results[svo]["base_runs"][0]
        ax.plot(base_run["consensus_level"], color=colors_svo[svo], linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(GOVERNANCE_PARAMS["consensus_threshold"], color='red', linestyle=':', alpha=0.7, label='Threshold')
    ax.set_title('Consensus Level (weighted agreement)', fontweight='bold')
    ax.set_xlabel('Round'); ax.set_ylabel('Agreement'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # 하좌: Baseline 입장 변화 (prosocial)
    ax = axes[1, 0]
    base_run = results["prosocial"]["base_runs"][0]
    for i, name in enumerate(sh_names):
        ax.plot(base_run["positions"][:, i], color=sh_colors[i], linewidth=2, label=name)
    ax.set_title('Stakeholder Positions (Prosocial Baseline)', fontweight='bold')
    ax.set_xlabel('Round'); ax.set_ylabel('Regulation Level')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # 하우: 복지 비교
    ax = axes[1, 1]
    for svo in SVO_CONDITIONS:
        meta_run = results[svo]["meta_runs"][0]
        ax.plot(meta_run["welfare"], color=colors_svo[svo], linewidth=2, label=f'{svo} (Meta)')
    ax.set_title('Social Welfare Over Time', fontweight='bold')
    ax.set_xlabel('Round'); ax.set_ylabel('Welfare'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, "fig43_governance.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[O3] Fig 43 저장: {path}")


def plot_fig44(results):
    """Fig 44: AI 거버넌스 ATE"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Fig 44: AI Governance ATE — Consensus Speed, Regulation, Welfare",
                 fontsize=14, fontweight='bold', y=1.02)
    
    svo_names = list(SVO_CONDITIONS.keys())
    x = np.arange(len(svo_names))
    labels = [s.capitalize() for s in svo_names]
    
    # ATE 합의 속도 (라운드 절약)
    ax = axes[0]
    ates = [results[s]["ate_consensus_speed"] for s in svo_names]
    colors_bar = ['#43a047' if v > 0 else '#e53935' for v in ates]
    ax.bar(x, ates, color=colors_bar, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_title('ATE: Rounds Saved (faster=positive)', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5); ax.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(ates):
        ax.text(i, v + 0.3 * np.sign(v), f'{v:+.1f}', ha='center', fontweight='bold', fontsize=9)
    
    # 최종 규제 수준
    ax = axes[1]
    meta_r = [results[s]["meta_regulation"] for s in svo_names]
    base_r = [results[s]["base_regulation"] for s in svo_names]
    ax.bar(x - 0.15, meta_r, 0.3, label='Meta ON', color='#1e88e5')
    ax.bar(x + 0.15, base_r, 0.3, label='Baseline', color='#90caf9')
    ax.set_title('Final Regulation Level', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=9)
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    
    # ATE 복지
    ax = axes[2]
    ates_w = [results[s]["ate_welfare"] for s in svo_names]
    colors_w = ['#43a047' if v > 0 else '#e53935' for v in ates_w]
    ax.bar(x, ates_w, color=colors_w, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_title('ATE: Social Welfare', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5); ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig44_governance_ate.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[O3] Fig 44 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [O3] AI 거버넌스 투표 게임")
    print("=" * 60)
    
    results = run_experiment()
    
    print("\n--- Governance Summary ---")
    for svo_name in SVO_CONDITIONS:
        d = results[svo_name]
        print(f"  {svo_name:>12s} | Meta ConsAt={d['meta_consensus_at']:.0f} | "
              f"Base={d['base_consensus_at']:.0f} | Saved={d['ate_consensus_speed']:+.1f}r | "
              f"Reg={d['meta_regulation']:.1f} | W={d['meta_welfare']:.3f}")
    
    plot_fig43(results)
    plot_fig44(results)
    
    json_data = {sn: {k: v for k, v in d.items() if k not in ("meta_runs", "base_runs")} for sn, d in results.items()}
    json_path = os.path.join(OUTPUT_DIR, "governance_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\n[O3] 결과 JSON: {json_path}")
