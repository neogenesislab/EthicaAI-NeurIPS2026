"""
O4~O6: 하이브리드 λ-LLM 에이전트 + 비교 실험
EthicaAI Phase O — 연구축 III: 에이전트 고도화

자원 수준에 따라 수학적 λ와 LLM 추론 중 최적 전략을 선택하는 라우터.
시뮬레이션 모드로 작동 (GPU 불필요).

출력: Fig 45 (성능 비교), Fig 46 (LLM 호출 분포)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import os
import sys
from enum import Enum

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "ETHICAAI_OUTPUT_DIR", "simulation/outputs/reproduce"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# === 설정 (하드코딩 금지) ===
ROUTER_CONFIG = {
    "ambiguity_low": 0.2,
    "ambiguity_high": 0.7,
    "llm_latency_ms": 50,
    "math_latency_ms": 0.1,
}

SVO_CONDITIONS = {"selfish": 0.0, "individualist": 15.0, "prosocial": 45.0, "altruistic": 90.0}
N_ROUNDS = 200
N_SEEDS = 10


class DecisionMode(Enum):
    MATH = "mathematical"
    LLM = "llm"
    HYBRID = "hybrid"


def compute_lambda_math(svo_theta_deg, resource):
    """수학적 λ 결정 (빠름, 결정적)"""
    svo_theta = np.radians(svo_theta_deg)
    lambda_base = np.sin(svo_theta)
    if resource < 0.2:
        return 0.0
    elif resource > 0.7:
        return min(1.0, lambda_base * 1.5)
    else:
        return lambda_base


def simulate_llm_decision(svo_theta_deg, resource, scenario_type, rng):
    """LLM 추론 시뮬레이션 (맥락적, 노이즈 포함)"""
    lambda_math = compute_lambda_math(svo_theta_deg, resource)
    
    # LLM은 맥락을 고려하여 미묘하게 다른 결정
    context_adjustments = {
        "standard": 0.0,
        "crisis": -0.05,      # 위기 시 더 보수적
        "abundance": +0.03,   # 풍요 시 더 관대
        "inequality": +0.08,  # 불평등 시 더 공정
        "competition": -0.03, # 경쟁 시 더 방어적
    }
    adjustment = context_adjustments.get(scenario_type, 0.0)
    llm_lambda = lambda_math + adjustment + rng.normal(0, 0.03)
    return np.clip(llm_lambda, 0, 1)


def hybrid_route(resource, mode):
    """라우터: 어떤 경로로 결정할지"""
    if mode == DecisionMode.MATH:
        return DecisionMode.MATH
    elif mode == DecisionMode.LLM:
        return DecisionMode.LLM
    else:  # HYBRID
        if resource < ROUTER_CONFIG["ambiguity_low"] or resource > ROUTER_CONFIG["ambiguity_high"]:
            return DecisionMode.MATH
        else:
            return DecisionMode.LLM


def simulate_agent(svo_theta_deg, seed, mode, llm_budget=50):
    """에이전트 시뮬레이션 (모드별)"""
    rng = np.random.RandomState(seed)
    
    resource = 0.5
    wealth = 100.0
    cooperation_history = []
    lambda_history = []
    llm_calls = 0
    latencies = []
    route_log = []
    
    scenarios = ["standard", "crisis", "abundance", "inequality", "competition"]
    
    for t in range(N_ROUNDS):
        scenario = scenarios[t % len(scenarios)]
        
        # 라우팅
        actual_mode = hybrid_route(resource, mode)
        
        # LLM 예산 초과 시 fallback
        if actual_mode == DecisionMode.LLM and llm_calls >= llm_budget:
            actual_mode = DecisionMode.MATH
        
        # 결정
        if actual_mode == DecisionMode.MATH:
            lambda_t = compute_lambda_math(svo_theta_deg, resource)
            latencies.append(ROUTER_CONFIG["math_latency_ms"])
        else:
            lambda_t = simulate_llm_decision(svo_theta_deg, resource, scenario, rng)
            latencies.append(ROUTER_CONFIG["llm_latency_ms"])
            llm_calls += 1
        
        route_log.append({"t": t, "resource": resource, "mode": actual_mode.value, "scenario": scenario})
        lambda_history.append(lambda_t)
        
        # 협력 결정
        cooperate = rng.random() < (0.3 + lambda_t * 0.5)
        cooperation_history.append(1 if cooperate else 0)
        
        # 환경 업데이트
        group_coop = 0.5 + lambda_t * 0.2
        if cooperate:
            resource = np.clip(resource + 0.02 * group_coop - 0.01, 0, 1)
            wealth += 2.0 * group_coop
        else:
            resource = np.clip(resource - 0.02, 0, 1)
            wealth += 3.0 if rng.random() > 0.5 else -1.0
    
    return {
        "cooperation_history": cooperation_history,
        "lambda_history": lambda_history,
        "llm_calls": llm_calls,
        "mean_latency": float(np.mean(latencies)),
        "total_latency": float(np.sum(latencies)),
        "final_coop": float(np.mean(cooperation_history[-30:])),
        "final_wealth": float(wealth),
        "route_log": route_log,
    }


def run_comparison():
    """5 조건 × 4 SVO × 10 seeds = 200 runs"""
    conditions = {
        "pure_math":   {"mode": DecisionMode.MATH, "budget": 0},
        "pure_llm":    {"mode": DecisionMode.LLM,  "budget": 999},
        "hybrid_20":   {"mode": DecisionMode.HYBRID, "budget": 20},
        "hybrid_50":   {"mode": DecisionMode.HYBRID, "budget": 50},
        "hybrid_100":  {"mode": DecisionMode.HYBRID, "budget": 100},
    }
    
    results = {}
    for cond_name, cond in conditions.items():
        results[cond_name] = {}
        for svo_name, svo_theta in SVO_CONDITIONS.items():
            runs = [simulate_agent(svo_theta, s, cond["mode"], cond["budget"]) for s in range(N_SEEDS)]
            results[cond_name][svo_name] = {
                "mean_coop": float(np.mean([r["final_coop"] for r in runs])),
                "mean_wealth": float(np.mean([r["final_wealth"] for r in runs])),
                "mean_latency": float(np.mean([r["mean_latency"] for r in runs])),
                "mean_llm_calls": float(np.mean([r["llm_calls"] for r in runs])),
                "runs": runs,
            }
    return results


def plot_fig45(results):
    """Fig 45: 하이브리드 에이전트 성능 비교"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Fig 45: Hybrid λ-LLM Agent — Performance Comparison",
                 fontsize=14, fontweight='bold', y=1.02)
    
    conditions = list(results.keys())
    cond_labels = ["Pure λ", "Pure LLM", "Hybrid-20", "Hybrid-50", "Hybrid-100"]
    x = np.arange(len(conditions))
    svo_colors = {'selfish': '#e53935', 'individualist': '#ff9800', 'prosocial': '#1e88e5', 'altruistic': '#43a047'}
    
    # 협력률
    ax = axes[0]
    width = 0.18
    for j, (svo_name, color) in enumerate(svo_colors.items()):
        vals = [results[c][svo_name]["mean_coop"] for c in conditions]
        ax.bar(x + (j - 1.5) * width, vals, width, label=svo_name.capitalize(), color=color, alpha=0.85)
    ax.set_title('Cooperation Rate', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(cond_labels, rotation=15, fontsize=8)
    ax.legend(fontsize=7); ax.grid(True, axis='y', alpha=0.3)
    
    # 복지
    ax = axes[1]
    for j, (svo_name, color) in enumerate(svo_colors.items()):
        vals = [results[c][svo_name]["mean_wealth"] for c in conditions]
        ax.bar(x + (j - 1.5) * width, vals, width, label=svo_name.capitalize(), color=color, alpha=0.85)
    ax.set_title('Final Wealth', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(cond_labels, rotation=15, fontsize=8)
    ax.legend(fontsize=7); ax.grid(True, axis='y', alpha=0.3)
    
    # 응답 속도
    ax = axes[2]
    latencies = [np.mean([results[c][s]["mean_latency"] for s in SVO_CONDITIONS]) for c in conditions]
    llm_calls = [np.mean([results[c][s]["mean_llm_calls"] for s in SVO_CONDITIONS]) for c in conditions]
    bar1 = ax.bar(x, latencies, 0.4, label='Avg Latency (ms)', color='#7e57c2', alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(x, llm_calls, 'o-', color='#ff5722', linewidth=2, markersize=8, label='LLM Calls')
    ax.set_title('Inference Cost', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(cond_labels, rotation=15, fontsize=8)
    ax.set_ylabel('Latency (ms)'); ax2.set_ylabel('LLM Calls', color='#ff5722')
    ax.legend(loc='upper left', fontsize=7); ax2.legend(loc='upper right', fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig45_hybrid_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[O6] Fig 45 저장: {path}")


def plot_fig46(results):
    """Fig 46: LLM 호출 분포 — 언제 LLM이 불리는가?"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fig 46: When Does the Hybrid Agent Call LLM?",
                 fontsize=14, fontweight='bold', y=1.02)
    
    # 좌: 자원 수준별 LLM 호출 빈도 (hybrid_50 prosocial)
    ax = axes[0]
    hybrid_data = results["hybrid_50"]["prosocial"]["runs"][0]
    llm_resources = [r["resource"] for r in hybrid_data["route_log"] if r["mode"] == "llm"]
    math_resources = [r["resource"] for r in hybrid_data["route_log"] if r["mode"] == "mathematical"]
    
    bins = np.linspace(0, 1, 20)
    ax.hist(llm_resources, bins=bins, alpha=0.7, color='#7e57c2', label=f'LLM ({len(llm_resources)} calls)', density=True)
    ax.hist(math_resources, bins=bins, alpha=0.5, color='#1e88e5', label=f'Math ({len(math_resources)} calls)', density=True)
    ax.axvline(ROUTER_CONFIG["ambiguity_low"], color='red', linestyle=':', alpha=0.7, label=f'Low={ROUTER_CONFIG["ambiguity_low"]}')
    ax.axvline(ROUTER_CONFIG["ambiguity_high"], color='red', linestyle=':', alpha=0.7, label=f'High={ROUTER_CONFIG["ambiguity_high"]}')
    ax.set_title('Resource Level Distribution by Route', fontweight='bold')
    ax.set_xlabel('Resource Level'); ax.set_ylabel('Density')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # 우: 시나리오별 LLM 호출 비율 (hybrid_50, 전 SVO)
    ax = axes[1]
    scenario_counts = {}
    total_counts = {}
    for svo_name in SVO_CONDITIONS:
        for run in results["hybrid_50"][svo_name]["runs"]:
            for entry in run["route_log"]:
                sc = entry["scenario"]
                total_counts[sc] = total_counts.get(sc, 0) + 1
                if entry["mode"] == "llm":
                    scenario_counts[sc] = scenario_counts.get(sc, 0) + 1
    
    scenarios = sorted(total_counts.keys())
    rates = [scenario_counts.get(s, 0) / (total_counts[s] + 1e-8) * 100 for s in scenarios]
    colors = plt.cm.Spectral(np.linspace(0.2, 0.8, len(scenarios)))
    ax.bar(scenarios, rates, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_title('LLM Call Rate by Scenario (%)', fontweight='bold')
    ax.set_ylabel('LLM Call Rate (%)')
    for i, v in enumerate(rates):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig46_llm_call_distribution.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[O6] Fig 46 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [O4~O6] 하이브리드 λ-LLM 에이전트")
    print("=" * 60)
    
    results = run_comparison()
    
    print("\n--- Hybrid Agent Comparison ---")
    print(f"{'Condition':>12s} | {'SVO':>12s} | {'Coop':>6s} | {'Wealth':>8s} | {'Latency':>8s} | {'LLM#':>5s}")
    print("-" * 65)
    for cond in results:
        for svo in SVO_CONDITIONS:
            d = results[cond][svo]
            print(f"{cond:>12s} | {svo:>12s} | {d['mean_coop']:.3f} | {d['mean_wealth']:8.1f} | "
                  f"{d['mean_latency']:7.2f}ms | {d['mean_llm_calls']:5.0f}")
        print("-" * 65)
    
    plot_fig45(results)
    plot_fig46(results)
    
    json_data = {cn: {sn: {k: v for k, v in sd.items() if k != "runs"}
                       for sn, sd in cd.items()} for cn, cd in results.items()}
    json_path = os.path.join(OUTPUT_DIR, "hybrid_agent_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\n[O6] 결과 JSON: {json_path}")
