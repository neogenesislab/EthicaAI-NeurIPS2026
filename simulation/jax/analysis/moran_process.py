"""
Q2: 진화 동역학 심화 — Moran Process
EthicaAI Phase Q — 유한 집단 확률 과정

무한 집단 Replicator Dynamics → 유한 집단 Moran Process 확장:
- 고정 확률(Fixation Probability) 계산
- 침입 저항성(Invasion Resistance) 분석
- 확률적 안정성(Stochastic Stability) 검증

출력: Fig 63 (Moran 고정 확률), Fig 64 (침입 저항성 히트맵)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json, os, sys

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "ETHICAAI_OUTPUT_DIR", "simulation/outputs/reproduce")
os.makedirs(OUTPUT_DIR, exist_ok=True)

POP_SIZES = [20, 50, 100]
N_SIMULATIONS = 5000
STRATEGIES = ["selfish", "individualist", "prosocial", "altruistic", "meta_ranking"]
ENDOWMENT = 100.0
SELECTION_STRENGTH = 1.0


def pgg_payoff(strategy, n_cooperators, n_total):
    """PGG 보수 함수"""
    contribution_map = {
        "selfish": 0.0, "individualist": 0.15, "prosocial": 0.55,
        "altruistic": 0.80, "meta_ranking": 0.60,
    }
    my_c = contribution_map[strategy] * ENDOWMENT
    avg_c = (n_cooperators * 0.55 * ENDOWMENT + (n_total - n_cooperators) * 0.1 * ENDOWMENT) / n_total
    public_good = (my_c + avg_c * (n_total - 1)) * 1.6 / n_total
    return (ENDOWMENT - my_c) + public_good


def moran_process(pop_size, invader, resident, n_sims=N_SIMULATIONS):
    """Moran 과정: 1명의 침입자가 고정될 확률"""
    fixation_count = 0
    
    for _ in range(n_sims):
        n_invader = 1
        
        for _ in range(pop_size * 100):
            if n_invader == 0:
                break
            if n_invader == pop_size:
                fixation_count += 1
                break
            
            n_resident = pop_size - n_invader
            fitness_inv = pgg_payoff(invader, n_invader, pop_size)
            fitness_res = pgg_payoff(resident, n_resident, pop_size)
            
            # 선택 확률 (Fermi 함수)
            prob_inv = 1 / (1 + np.exp(-SELECTION_STRENGTH * (fitness_inv - fitness_res)))
            
            if np.random.random() < prob_inv:
                n_invader += 1
            else:
                n_invader -= 1
    
    return fixation_count / n_sims


def run_fixation_analysis():
    """모든 전략 쌍의 고정 확률 계산"""
    results = {}
    for pop_size in POP_SIZES:
        neutral = 1.0 / pop_size
        results[pop_size] = {"neutral": neutral, "matrix": {}}
        
        for invader in STRATEGIES:
            results[pop_size]["matrix"][invader] = {}
            for resident in STRATEGIES:
                if invader == resident:
                    results[pop_size]["matrix"][invader][resident] = None
                    continue
                fix_prob = moran_process(pop_size, invader, resident, N_SIMULATIONS)
                results[pop_size]["matrix"][invader][resident] = float(fix_prob)
    
    return results


def plot_fig63(results):
    fig, axes = plt.subplots(1, len(POP_SIZES), figsize=(6*len(POP_SIZES), 5.5))
    fig.suptitle("Fig 63: Moran Process — Fixation Probabilities Across Population Sizes",
                 fontsize=14, fontweight='bold', y=1.02)
    
    for idx, n in enumerate(POP_SIZES):
        ax = axes[idx]
        matrix = np.zeros((len(STRATEGIES), len(STRATEGIES)))
        neutral = results[n]["neutral"]
        
        for i, inv in enumerate(STRATEGIES):
            for j, res in enumerate(STRATEGIES):
                val = results[n]["matrix"][inv][res]
                matrix[i, j] = val if val is not None else 0
        
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=max(0.2, np.max(matrix)))
        
        for i in range(len(STRATEGIES)):
            for j in range(len(STRATEGIES)):
                val = results[n]["matrix"][STRATEGIES[i]][STRATEGIES[j]]
                if val is not None:
                    color = 'white' if val < 0.05 else 'black'
                    marker = '▲' if val > neutral * 1.5 else '▼' if val < neutral * 0.5 else '●'
                    ax.text(j, i, f'{val:.3f}\n{marker}', ha='center', va='center', 
                           fontsize=7, fontweight='bold', color=color)
                else:
                    ax.text(j, i, '—', ha='center', va='center', fontsize=10, color='grey')
        
        labels = [s[:5] for s in STRATEGIES]
        ax.set_xticks(range(len(STRATEGIES))); ax.set_xticklabels(labels, fontsize=7, rotation=30)
        ax.set_yticks(range(len(STRATEGIES))); ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f'N = {n} (neutral = {neutral:.3f})', fontweight='bold')
        ax.set_xlabel('Resident'); ax.set_ylabel('Invader')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig63_moran_fixation.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[Q2] Fig 63 저장: {path}")


def plot_fig64(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Fig 64: Invasion Resistance & Evolutionary Dominance",
                 fontsize=14, fontweight='bold', y=1.02)
    
    colors = {'selfish': '#e53935', 'individualist': '#ff9800', 'prosocial': '#1e88e5',
              'altruistic': '#43a047', 'meta_ranking': '#7e57c2'}
    
    # 좌: 침입 저항성 (방어 성공률)
    ax = axes[0]
    for pop_size in POP_SIZES:
        defense_rates = []
        neutral = results[pop_size]["neutral"]
        for resident in STRATEGIES:
            invasions = [results[pop_size]["matrix"][inv][resident] 
                        for inv in STRATEGIES if inv != resident and results[pop_size]["matrix"][inv][resident] is not None]
            defense = 1 - np.mean(invasions) / neutral if neutral > 0 else 0
            defense_rates.append(max(0, float(defense)))
        ax.plot(range(len(STRATEGIES)), defense_rates, 'o-', linewidth=2, markersize=8, label=f'N={pop_size}')
    
    ax.set_xticks(range(len(STRATEGIES))); ax.set_xticklabels([s[:6] for s in STRATEGIES])
    ax.set_title('Invasion Resistance by Strategy', fontweight='bold')
    ax.set_ylabel('Defense Rate (1 - avg_fix/neutral)')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # 우: Meta-Ranking 침입 성공률
    ax = axes[1]
    x = np.arange(len(STRATEGIES) - 1)
    others = [s for s in STRATEGIES if s != "meta_ranking"]
    for pop_size in POP_SIZES:
        fix_probs = [results[pop_size]["matrix"]["meta_ranking"][res] for res in others]
        neutral = results[pop_size]["neutral"]
        ax.bar(x + POP_SIZES.index(pop_size) * 0.25, fix_probs, 0.23, label=f'N={pop_size}', alpha=0.8)
    
    ax.axhline(results[POP_SIZES[0]]["neutral"], color='red', linestyle='--', label='Neutral drift')
    ax.set_xticks(x + 0.25); ax.set_xticklabels([s[:6] for s in others])
    ax.set_title('Meta-Ranking Fixation Against Others', fontweight='bold')
    ax.set_ylabel('Fixation Probability'); ax.legend(fontsize=7); ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig64_invasion_resistance.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[Q2] Fig 64 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [Q2] Moran Process 진화 동역학")
    print("=" * 60)
    results = run_fixation_analysis()
    for n in POP_SIZES:
        print(f"\n  N={n} (neutral={results[n]['neutral']:.4f}):")
        print(f"    Meta→selfish: {results[n]['matrix']['meta_ranking']['selfish']:.4f}")
        print(f"    Meta→prosocial: {results[n]['matrix']['meta_ranking']['prosocial']:.4f}")
        print(f"    Selfish→meta: {results[n]['matrix']['selfish']['meta_ranking']:.4f}")
    plot_fig63(results)
    plot_fig64(results)
    json_path = os.path.join(OUTPUT_DIR, "moran_results.json")
    with open(json_path, 'w') as f:
        json.dump({str(n): {"neutral": r["neutral"], 
                            "matrix": {i: {j: v for j, v in jd.items()} for i, jd in r["matrix"].items()}}
                  for n, r in results.items()}, f, indent=2)
    print(f"\n[Q2] JSON: {json_path}")
