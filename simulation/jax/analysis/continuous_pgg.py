"""
P3: 연속 상태/행동 공간 PGG (Continuous-Space PGG)
EthicaAI Phase P — 환경 현실성 보강

Grid World를 넘어 연속 기여량 결정 환경에서 메타랭킹 효과 검증.
- 연속 행동: 기여량 ∈ [0, endowment] (Beta 분포 정책)
- 연속 상태: 자원 R ∈ [0,1], 공공재 수준 G ∈ [0,∞)
- 비선형 생산함수: G = A·(ΣC)^α (α<1, 수확 체감)

출력: Fig 57 (연속 PGG 결과), Fig 58 (비선형 생산함수 효과)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json, os, sys

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "ETHICAAI_OUTPUT_DIR", "simulation/outputs/reproduce")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_AGENTS = 50
N_STEPS = 300
N_SEEDS = 10
ENDOWMENT = 100.0
ALPHAS = [0.5, 0.7, 1.0, 1.3]  # 비선형 생산함수 지수
A_PROD = 2.0  # 생산 계수


def beta_policy(lambda_t, rng):
    """Beta 분포 기반 연속 기여 정책"""
    a = max(0.1, lambda_t * 5)
    b = max(0.1, (1 - lambda_t) * 5)
    return rng.beta(a, b) * ENDOWMENT


def meta_lambda(svo_deg, resource, prev_lambda):
    """메타랭킹 동적 λ"""
    base = np.sin(np.radians(svo_deg))
    if resource < 0.2:
        target = max(0, base * 0.3)
    elif resource > 0.7:
        target = min(1, base * 1.5)
    else:
        target = base
    return 0.9 * prev_lambda + 0.1 * target


def simulate_continuous(svo_deg, alpha, seed, use_meta=True):
    rng = np.random.RandomState(seed)
    lambdas = np.full(N_AGENTS, np.sin(np.radians(svo_deg)))
    resource = 0.5
    
    contrib_history, welfare_history, resource_history = [], [], []
    
    for t in range(N_STEPS):
        if use_meta:
            lambdas = np.array([meta_lambda(svo_deg, resource, l) for l in lambdas])
        
        contribs = np.array([beta_policy(l, rng) for l in lambdas])
        total_c = np.sum(contribs)
        
        public_good = A_PROD * (total_c ** alpha) / N_AGENTS
        payoffs = (ENDOWMENT - contribs) + public_good
        welfare = np.mean(payoffs)
        
        mean_rate = np.mean(contribs) / ENDOWMENT
        resource = np.clip(resource + 0.02 * (mean_rate - 0.3), 0, 1)
        
        contrib_history.append(float(np.mean(contribs)))
        welfare_history.append(float(welfare))
        resource_history.append(float(resource))
    
    return {
        "mean_contrib": float(np.mean(contrib_history[-30:])),
        "mean_welfare": float(np.mean(welfare_history[-30:])),
        "mean_resource": float(np.mean(resource_history[-30:])),
        "contrib_history": contrib_history,
        "welfare_history": welfare_history,
    }


def run_experiment():
    svos = {"selfish": 0.0, "individualist": 15.0, "prosocial": 45.0, "altruistic": 90.0}
    results = {}
    
    for alpha in ALPHAS:
        results[alpha] = {}
        for svo_name, svo_deg in svos.items():
            meta_runs = [simulate_continuous(svo_deg, alpha, s, True) for s in range(N_SEEDS)]
            base_runs = [simulate_continuous(svo_deg, alpha, s, False) for s in range(N_SEEDS)]
            
            results[alpha][svo_name] = {
                "meta_welfare": float(np.mean([r["mean_welfare"] for r in meta_runs])),
                "base_welfare": float(np.mean([r["mean_welfare"] for r in base_runs])),
                "ate_welfare": float(np.mean([r["mean_welfare"] for r in meta_runs]) - 
                                    np.mean([r["mean_welfare"] for r in base_runs])),
                "meta_contrib": float(np.mean([r["mean_contrib"] for r in meta_runs])),
                "base_contrib": float(np.mean([r["mean_contrib"] for r in base_runs])),
                "meta_history": [float(np.mean([r["welfare_history"][t] for r in meta_runs])) for t in range(N_STEPS)],
                "base_history": [float(np.mean([r["welfare_history"][t] for r in base_runs])) for t in range(N_STEPS)],
            }
    return results


def plot_fig57(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fig 57: Continuous-Space PGG — Meta-Ranking with Nonlinear Production",
                 fontsize=14, fontweight='bold', y=0.98)
    colors = {'selfish': '#e53935', 'individualist': '#ff9800', 'prosocial': '#1e88e5', 'altruistic': '#43a047'}
    
    for idx, alpha in enumerate(ALPHAS):
        ax = axes[idx // 2, idx % 2]
        for svo, color in colors.items():
            d = results[alpha][svo]
            ax.plot(d["meta_history"], color=color, linewidth=1.5, label=f'{svo[:5]} Meta')
            ax.plot(d["base_history"], color=color, linewidth=1, alpha=0.4, linestyle='--')
        ax.set_title(f'α = {alpha} ({"diminishing" if alpha < 1 else "linear" if alpha == 1 else "increasing"})',
                    fontweight='bold')
        ax.set_xlabel('Step'); ax.set_ylabel('Welfare')
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, "fig57_continuous_pgg.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[P3] Fig 57 저장: {path}")


def plot_fig58(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Fig 58: Nonlinear Production Function Effects on ATE",
                 fontsize=14, fontweight='bold', y=1.02)
    
    svos = list(results[ALPHAS[0]].keys())
    colors = {'selfish': '#e53935', 'individualist': '#ff9800', 'prosocial': '#1e88e5', 'altruistic': '#43a047'}
    
    ax = axes[0]
    for svo in svos:
        ates = [results[a][svo]["ate_welfare"] for a in ALPHAS]
        ax.plot(ALPHAS, ates, 'o-', color=colors[svo], linewidth=2, label=svo.capitalize())
    ax.axhline(0, color='grey', linestyle=':', alpha=0.5)
    ax.set_title('ATE(Welfare) by Production Exponent', fontweight='bold')
    ax.set_xlabel('α (Production Exponent)'); ax.set_ylabel('ATE'); ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    x = np.arange(len(svos))
    for i, alpha in enumerate(ALPHAS):
        vals = [results[alpha][s]["meta_welfare"] for s in svos]
        ax.bar(x + i * 0.2, vals, 0.18, label=f'α={alpha}', alpha=0.8)
    ax.set_title('Meta-Ranking Welfare by α', fontweight='bold')
    ax.set_xticks(x + 0.3); ax.set_xticklabels([s[:5] for s in svos])
    ax.legend(fontsize=7); ax.grid(True, axis='y', alpha=0.3)
    
    ax = axes[2]
    for svo in svos:
        meta_c = [results[a][svo]["meta_contrib"] for a in ALPHAS]
        ax.plot(ALPHAS, meta_c, 'o-', color=colors[svo], linewidth=2, label=svo.capitalize())
    ax.set_title('Mean Contribution by α', fontweight='bold')
    ax.set_xlabel('α'); ax.set_ylabel('Contribution'); ax.legend(); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig58_nonlinear_production.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[P3] Fig 58 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [P3] 연속 공간 PGG + 비선형 생산함수")
    print("=" * 60)
    results = run_experiment()
    for a in ALPHAS:
        print(f"\n  α={a}:")
        for s in results[a]:
            d = results[a][s]
            print(f"    {s:>12}: Meta W={d['meta_welfare']:.1f}, Base W={d['base_welfare']:.1f}, ATE={d['ate_welfare']:+.1f}")
    plot_fig57(results)
    plot_fig58(results)
    json_path = os.path.join(OUTPUT_DIR, "continuous_pgg_results.json")
    with open(json_path, 'w') as f:
        json.dump({str(a): {s: {k: v for k, v in d.items() if 'history' not in k} 
                            for s, d in sd.items()} for a, sd in results.items()}, f, indent=2)
    print(f"\n[P3] JSON: {json_path}")
