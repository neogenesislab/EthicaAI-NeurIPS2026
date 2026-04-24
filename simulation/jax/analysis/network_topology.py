"""
P4: 네트워크 토폴로지 효과
EthicaAI Phase P — 구조적 다양성 테스트

메타랭킹의 효과가 네트워크 구조에 따라 어떻게 달라지는지 검증:
- Complete: 모든 에이전트 연결
- Small-World (Watts-Strogatz): 클러스터링 + 짧은 경로
- Scale-Free (Barabási-Albert): 허브 중심 구조
- Ring Lattice: 최소 연결
- Random (Erdős-Rényi): 무작위 연결

출력: Fig 59 (토폴로지별 협력), Fig 60 (정보 전파 속도)
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
K_NEIGHBORS = 6


def generate_network(topology, n, seed=0):
    """인접 행렬 생성"""
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n))
    
    if topology == "complete":
        adj = np.ones((n, n)) - np.eye(n)
    elif topology == "ring":
        for i in range(n):
            for k in range(1, K_NEIGHBORS // 2 + 1):
                adj[i, (i + k) % n] = adj[(i + k) % n, i] = 1
    elif topology == "small_world":
        for i in range(n):
            for k in range(1, K_NEIGHBORS // 2 + 1):
                adj[i, (i + k) % n] = adj[(i + k) % n, i] = 1
        p_rewire = 0.1
        for i in range(n):
            for k in range(1, K_NEIGHBORS // 2 + 1):
                if rng.random() < p_rewire:
                    j = (i + k) % n
                    adj[i, j] = adj[j, i] = 0
                    new_j = rng.randint(n)
                    while new_j == i or adj[i, new_j] == 1:
                        new_j = rng.randint(n)
                    adj[i, new_j] = adj[new_j, i] = 1
    elif topology == "scale_free":
        m = K_NEIGHBORS // 2
        adj[:m, :m] = 1 - np.eye(m)
        degrees = np.sum(adj, axis=1)
        for i in range(m, n):
            probs = degrees[:i] / (np.sum(degrees[:i]) + 1e-10)
            targets = rng.choice(i, size=min(m, i), replace=False, p=probs)
            for t in targets:
                adj[i, t] = adj[t, i] = 1
            degrees = np.sum(adj, axis=1)
    elif topology == "random":
        p = K_NEIGHBORS / (n - 1)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[i, j] = adj[j, i] = 1
    return adj


def simulate_network(topology, svo_deg, seed, use_meta=True):
    rng = np.random.RandomState(seed)
    adj = generate_network(topology, N_AGENTS, seed)
    lambdas = np.full(N_AGENTS, np.sin(np.radians(svo_deg)))
    resource = 0.5
    
    coop_hist, welfare_hist, spread_hist = [], [], []
    
    for t in range(N_STEPS):
        if use_meta:
            new_lambdas = np.copy(lambdas)
            for i in range(N_AGENTS):
                neighbors = np.where(adj[i] > 0)[0]
                if len(neighbors) > 0:
                    neighbor_avg = np.mean(lambdas[neighbors])
                else:
                    neighbor_avg = lambdas[i]
                
                base = np.sin(np.radians(svo_deg))
                if resource < 0.2:
                    target = max(0, base * 0.3)
                elif resource > 0.7:
                    target = min(1, base * 1.5)
                else:
                    target = base * (0.5 + 0.5 * neighbor_avg)
                new_lambdas[i] = 0.9 * lambdas[i] + 0.1 * target
            lambdas = new_lambdas
        
        contribs = np.clip(lambdas * ENDOWMENT * 0.8 + rng.normal(0, 3, N_AGENTS), 0, ENDOWMENT)
        total = np.sum(contribs)
        public_good = total * 1.6 / N_AGENTS
        payoffs = (ENDOWMENT - contribs) + public_good
        
        resource = np.clip(resource + 0.02 * (np.mean(contribs) / ENDOWMENT - 0.3), 0, 1)
        
        coop_hist.append(float(np.mean(contribs > ENDOWMENT * 0.3)))
        welfare_hist.append(float(np.mean(payoffs)))
        spread_hist.append(float(np.std(lambdas)))
    
    return {
        "mean_coop": float(np.mean(coop_hist[-30:])),
        "mean_welfare": float(np.mean(welfare_hist[-30:])),
        "lambda_spread": float(np.mean(spread_hist[-30:])),
        "convergence_speed": _convergence_speed(spread_hist),
        "coop_history": coop_hist,
        "welfare_history": welfare_hist,
    }


def _convergence_speed(spread_hist):
    init = spread_hist[0] if spread_hist[0] > 0 else 1e-8
    for t, s in enumerate(spread_hist):
        if s < init * 0.1:
            return t
    return N_STEPS


TOPOLOGIES = ["complete", "small_world", "scale_free", "ring", "random"]


def run_experiment():
    results = {}
    for topo in TOPOLOGIES:
        meta_runs = [simulate_network(topo, 45.0, s, True) for s in range(N_SEEDS)]
        base_runs = [simulate_network(topo, 45.0, s, False) for s in range(N_SEEDS)]
        results[topo] = {
            "meta_coop": float(np.mean([r["mean_coop"] for r in meta_runs])),
            "base_coop": float(np.mean([r["mean_coop"] for r in base_runs])),
            "ate": float(np.mean([r["mean_coop"] for r in meta_runs]) - np.mean([r["mean_coop"] for r in base_runs])),
            "meta_welfare": float(np.mean([r["mean_welfare"] for r in meta_runs])),
            "convergence": float(np.mean([r["convergence_speed"] for r in meta_runs])),
            "lambda_spread": float(np.mean([r["lambda_spread"] for r in meta_runs])),
            "meta_history": [float(np.mean([r["coop_history"][t] for r in meta_runs])) for t in range(N_STEPS)],
            "base_history": [float(np.mean([r["coop_history"][t] for r in base_runs])) for t in range(N_STEPS)],
        }
    return results


def plot_fig59(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Fig 59: Network Topology Effects on Meta-Ranking Cooperation",
                 fontsize=14, fontweight='bold', y=1.02)
    topo_colors = {'complete': '#1e88e5', 'small_world': '#43a047', 'scale_free': '#ff9800',
                   'ring': '#e53935', 'random': '#7e57c2'}
    
    ax = axes[0]
    for topo, color in topo_colors.items():
        ax.plot(results[topo]["meta_history"], color=color, linewidth=2, label=topo.replace('_', ' ').title())
        ax.plot(results[topo]["base_history"], color=color, linewidth=1, alpha=0.3, linestyle='--')
    ax.set_title('Cooperation Over Time (solid=Meta, dashed=Base)', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Cooperation Rate'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    x = np.arange(len(TOPOLOGIES))
    meta_vals = [results[t]["meta_coop"] for t in TOPOLOGIES]
    base_vals = [results[t]["base_coop"] for t in TOPOLOGIES]
    ax.bar(x - 0.2, meta_vals, 0.35, label='Meta', color='#1e88e5', alpha=0.8)
    ax.bar(x + 0.2, base_vals, 0.35, label='Baseline', color='#e53935', alpha=0.8)
    ax.set_title('Final Cooperation by Topology', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([t[:8] for t in TOPOLOGIES], rotation=15)
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    
    ax = axes[2]
    ates = [results[t]["ate"] for t in TOPOLOGIES]
    bars = ax.bar(x, ates, color=[topo_colors[t] for t in TOPOLOGIES], alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, ates):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.002, f'{val:+.3f}', ha='center', fontsize=9)
    ax.axhline(0, color='grey', linestyle=':')
    ax.set_title('ATE by Topology', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([t[:8] for t in TOPOLOGIES], rotation=15)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig59_network_topology.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[P4] Fig 59 저장: {path}")


def plot_fig60(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Fig 60: Information Propagation & λ Convergence by Topology",
                 fontsize=14, fontweight='bold', y=1.02)
    topo_colors = {'complete': '#1e88e5', 'small_world': '#43a047', 'scale_free': '#ff9800',
                   'ring': '#e53935', 'random': '#7e57c2'}
    
    ax = axes[0]
    conv = [results[t]["convergence"] for t in TOPOLOGIES]
    bars = ax.bar(range(len(TOPOLOGIES)), conv, color=[topo_colors[t] for t in TOPOLOGIES], alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, conv):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.0f}', ha='center', fontweight='bold')
    ax.set_title('λ Convergence Speed (steps to 90%)', fontweight='bold')
    ax.set_xticks(range(len(TOPOLOGIES))); ax.set_xticklabels([t[:8] for t in TOPOLOGIES], rotation=15)
    ax.set_ylabel('Steps'); ax.grid(True, axis='y', alpha=0.3)
    
    ax = axes[1]
    spreads = [results[t]["lambda_spread"] for t in TOPOLOGIES]
    welfares = [results[t]["meta_welfare"] for t in TOPOLOGIES]
    for i, topo in enumerate(TOPOLOGIES):
        ax.scatter(spreads[i], welfares[i], c=topo_colors[topo], s=200, zorder=5,
                  edgecolors='black', linewidth=1.5, label=topo.replace('_', ' ').title())
    ax.set_title('λ Spread vs Welfare (Topology Trade-off)', fontweight='bold')
    ax.set_xlabel('λ Spread (Std)'); ax.set_ylabel('Welfare')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig60_information_propagation.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[P4] Fig 60 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [P4] 네트워크 토폴로지 효과")
    print("=" * 60)
    results = run_experiment()
    print(f"\n{'Topology':>12} | {'Meta':>5} | {'Base':>5} | {'ATE':>7} | {'Conv':>5} | {'Spread':>6}")
    print("-" * 55)
    for topo in TOPOLOGIES:
        d = results[topo]
        print(f"{topo:>12} | {d['meta_coop']:.3f} | {d['base_coop']:.3f} | {d['ate']:+.4f} | {d['convergence']:5.0f} | {d['lambda_spread']:.4f}")
    plot_fig59(results)
    plot_fig60(results)
    json_path = os.path.join(OUTPUT_DIR, "network_topology_results.json")
    with open(json_path, 'w') as f:
        json.dump({t: {k: v for k, v in d.items() if 'history' not in k} for t, d in results.items()}, f, indent=2)
    print(f"\n[P4] JSON: {json_path}")
