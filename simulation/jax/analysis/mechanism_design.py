"""
P5: Mechanism Design 이론 연결 — Incentive Compatibility 분석
EthicaAI Phase P — 이론 강화

메타랭킹의 게임 이론적 속성 분석:
1. Incentive Compatibility (IC): 정직 보고가 최적인지
2. Individual Rationality (IR): 참여가 비참여보다 좋은지
3. Budget Balance: 메커니즘이 재정적으로 균형인지
4. Nash Equilibrium 분석: λ* 프로파일의 안정성

출력: Fig 61 (IC/IR 검증), Fig 62 (Nash 균형 경관)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json, os, sys

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "ETHICAAI_OUTPUT_DIR", "simulation/outputs/reproduce")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_AGENTS = 20
ENDOWMENT = 100.0
MULTIPLIER = 1.6  # 공공재 배율


def payoff(contribs, i):
    """에이전트 i의 보수"""
    total = np.sum(contribs)
    public_good = MULTIPLIER * total / N_AGENTS
    return (ENDOWMENT - contribs[i]) + public_good


def meta_contribution(svo_deg, resource):
    base = np.sin(np.radians(svo_deg))
    if resource < 0.2:
        target = max(0, base * 0.3)
    elif resource > 0.7:
        target = min(1, base * 1.5)
    else:
        target = base
    return target * ENDOWMENT * 0.8


def test_incentive_compatibility(svo_degs, resource_levels):
    """IC 테스트: 정직 SVO 보고 vs 과소/과대 보고 비교"""
    results = []
    for svo in svo_degs:
        for resource in resource_levels:
            honest_contrib = meta_contribution(svo, resource)
            
            contribs_honest = np.full(N_AGENTS, honest_contrib)
            honest_payoff = payoff(contribs_honest, 0)
            
            deviations = np.linspace(0, ENDOWMENT, 50)
            dev_payoffs = []
            for dev in deviations:
                contribs_dev = np.full(N_AGENTS, honest_contrib)
                contribs_dev[0] = dev
                dev_payoffs.append(payoff(contribs_dev, 0))
            
            best_dev = deviations[np.argmax(dev_payoffs)]
            ic_satisfied = abs(best_dev - honest_contrib) < ENDOWMENT * 0.15
            
            results.append({
                "svo": float(svo),
                "resource": float(resource),
                "honest_contrib": float(honest_contrib),
                "honest_payoff": float(honest_payoff),
                "best_deviation": float(best_dev),
                "best_dev_payoff": float(max(dev_payoffs)),
                "ic_gap": float(max(dev_payoffs) - honest_payoff),
                "ic_satisfied": bool(ic_satisfied),
                "deviations": deviations.tolist(),
                "dev_payoffs": dev_payoffs,
            })
    return results


def test_individual_rationality():
    """IR 테스트: 메타랭킹 참여 vs 무참여 보수 비교"""
    svo_range = np.linspace(0, 90, 19)
    results = []
    for svo in svo_range:
        for resource in [0.1, 0.3, 0.5, 0.7, 0.9]:
            meta_c = meta_contribution(svo, resource)
            
            contribs_participate = np.full(N_AGENTS, meta_c)
            participate_payoff = payoff(contribs_participate, 0)
            
            contribs_free = np.full(N_AGENTS, meta_c)
            contribs_free[0] = 0
            free_ride_payoff = payoff(contribs_free, 0)
            
            ir_satisfied = participate_payoff >= ENDOWMENT
            
            results.append({
                "svo": float(svo),
                "resource": float(resource),
                "participate_payoff": float(participate_payoff),
                "free_ride_payoff": float(free_ride_payoff),
                "standalone_payoff": float(ENDOWMENT),
                "ir_satisfied": bool(ir_satisfied),
            })
    return results


def nash_equilibrium_landscape():
    """Nash 균형 경관: λ에 대한 Best Response 분석"""
    lambda_range = np.linspace(0, 1, 50)
    br_landscape = np.zeros((50, 50))
    
    for i, lambda_others in enumerate(lambda_range):
        others_contrib = lambda_others * ENDOWMENT * 0.8
        contribs = np.full(N_AGENTS, others_contrib)
        
        for j, my_lambda in enumerate(lambda_range):
            contribs[0] = my_lambda * ENDOWMENT * 0.8
            br_landscape[j, i] = payoff(contribs, 0)
    
    best_responses = lambda_range[np.argmax(br_landscape, axis=0)]
    nash_candidates = []
    for i, (lr, br) in enumerate(zip(lambda_range, best_responses)):
        if abs(lr - br) < 0.05:
            nash_candidates.append((float(lr), float(br_landscape[np.argmax(br_landscape[:, i]), i])))
    
    return lambda_range, br_landscape, best_responses, nash_candidates


def plot_fig61(ic_results, ir_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Fig 61: Mechanism Design Properties — IC & IR Verification",
                 fontsize=14, fontweight='bold', y=1.02)
    
    ax = axes[0]
    for r in ic_results:
        if r["resource"] == 0.5:
            ax.plot(r["deviations"], r["dev_payoffs"], alpha=0.6, linewidth=1.5)
            ax.axvline(r["honest_contrib"], color='green', linestyle=':', alpha=0.3)
    ax.set_title('Payoff vs Deviation (R=0.5)', fontweight='bold')
    ax.set_xlabel('Contribution (deviation)'); ax.set_ylabel('Payoff')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    svos = sorted(set(r["svo"] for r in ic_results))
    resources = sorted(set(r["resource"] for r in ic_results))
    ic_matrix = np.zeros((len(svos), len(resources)))
    for r in ic_results:
        i = svos.index(r["svo"])
        j = resources.index(r["resource"])
        ic_matrix[i, j] = r["ic_gap"]
    im = ax.imshow(ic_matrix, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(resources))); ax.set_xticklabels([f'{r:.1f}' for r in resources], fontsize=7)
    ax.set_yticks(range(len(svos))); ax.set_yticklabels([f'{s:.0f}°' for s in svos], fontsize=7)
    ax.set_title('IC Gap (SVO × Resource)', fontweight='bold')
    ax.set_xlabel('Resource Level'); ax.set_ylabel('SVO θ')
    plt.colorbar(im, ax=ax, label='IC Gap (deviation gain)')
    
    ax = axes[2]
    svo_vals = [r["svo"] for r in ir_results if r["resource"] == 0.5]
    part_payoffs = [r["participate_payoff"] for r in ir_results if r["resource"] == 0.5]
    free_payoffs = [r["free_ride_payoff"] for r in ir_results if r["resource"] == 0.5]
    ax.plot(svo_vals, part_payoffs, 'o-', color='#1e88e5', linewidth=2, label='Participate (Meta)')
    ax.plot(svo_vals, free_payoffs, 's-', color='#e53935', linewidth=2, label='Free-Ride')
    ax.axhline(ENDOWMENT, color='grey', linestyle=':', label='Standalone (100)')
    ax.set_title('Individual Rationality (R=0.5)', fontweight='bold')
    ax.set_xlabel('SVO θ (degrees)'); ax.set_ylabel('Payoff')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig61_mechanism_design.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[P5] Fig 61 저장: {path}")


def plot_fig62(lambda_range, br_landscape, best_responses, nash_candidates):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Fig 62: Nash Equilibrium Landscape of Meta-Ranking",
                 fontsize=14, fontweight='bold', y=1.02)
    
    ax = axes[0]
    im = ax.imshow(br_landscape, origin='lower', cmap='viridis', aspect='auto',
                   extent=[0, 1, 0, 1])
    ax.plot(lambda_range, best_responses, 'r-', linewidth=2.5, label='Best Response BR(λ*)')
    ax.plot([0, 1], [0, 1], 'w--', linewidth=1.5, alpha=0.5, label='45° line (NE)')
    for nc in nash_candidates:
        ax.plot(nc[0], nc[0], 'w*', markersize=15, markeredgecolor='red', markeredgewidth=2)
    ax.set_title('Best Response Landscape', fontweight='bold')
    ax.set_xlabel('Others\' λ*'); ax.set_ylabel('My λ')
    ax.legend(fontsize=8, loc='upper left'); plt.colorbar(im, ax=ax, label='Payoff')
    
    ax = axes[1]
    ax.plot(lambda_range, best_responses, 'r-', linewidth=2.5, label='BR(λ*)')
    ax.plot(lambda_range, lambda_range, 'b--', linewidth=1.5, label='45° (λ=λ*)')
    ax.fill_between(lambda_range, best_responses, lambda_range, alpha=0.1,
                    color='red', where=best_responses < lambda_range)
    ax.fill_between(lambda_range, best_responses, lambda_range, alpha=0.1,
                    color='blue', where=best_responses >= lambda_range)
    for nc in nash_candidates:
        ax.plot(nc[0], nc[0], 'r*', markersize=20, label=f'NE at λ≈{nc[0]:.2f}')
    ax.set_title('Nash Equilibrium Analysis', fontweight='bold')
    ax.set_xlabel('Others\' λ*'); ax.set_ylabel('Best Response λ')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig62_nash_equilibrium.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[P5] Fig 62 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [P5] Mechanism Design 이론 연결")
    print("=" * 60)
    
    svos = [0, 15, 30, 45, 60, 75, 90]
    resources = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print("\n[1/3] Incentive Compatibility 테스트...")
    ic_results = test_incentive_compatibility(svos, resources)
    ic_count = sum(1 for r in ic_results if r["ic_satisfied"])
    print(f"  IC 충족: {ic_count}/{len(ic_results)} ({ic_count/len(ic_results)*100:.1f}%)")
    
    print("\n[2/3] Individual Rationality 테스트...")
    ir_results = test_individual_rationality()
    ir_count = sum(1 for r in ir_results if r["ir_satisfied"])
    print(f"  IR 충족: {ir_count}/{len(ir_results)} ({ir_count/len(ir_results)*100:.1f}%)")
    
    print("\n[3/3] Nash Equilibrium 분석...")
    lr, brl, br, nash = nash_equilibrium_landscape()
    print(f"  Nash 균형점: {len(nash)}개")
    for nc in nash:
        print(f"    λ* ≈ {nc[0]:.2f}, Payoff = {nc[1]:.1f}")
    
    plot_fig61(ic_results, ir_results)
    plot_fig62(lr, brl, br, nash)
    
    json_data = {
        "ic_rate": ic_count / len(ic_results),
        "ir_rate": ir_count / len(ir_results),
        "nash_equilibria": nash,
    }
    json_path = os.path.join(OUTPUT_DIR, "mechanism_design_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\n[P5] JSON: {json_path}")
