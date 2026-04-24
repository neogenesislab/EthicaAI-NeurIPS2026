"""
Q4: GNN 에이전트 — 이웃 정보 그래프 인코딩
EthicaAI Phase Q — 에이전트 아키텍처 확장

Graph Attention 기반 λ_t 결정:
- 이웃 에이전트의 (SVO, λ, contribution) 정보를 그래프로 처리
- Attention weight로 어떤 이웃에 가장 영향받는지 분석
- 단순 평균 대비 GNN 에이전트의 성능 차이 측정

출력: Fig 69 (GNN vs 평균 비교), Fig 70 (Attention 분석)
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
N_STEPS = 200
N_SEEDS = 10
ENDOWMENT = 100.0
K = 6  # 이웃 수


def build_ring_graph(n, k):
    adj = np.zeros((n, n))
    for i in range(n):
        for d in range(1, k // 2 + 1):
            adj[i, (i + d) % n] = 1
            adj[(i + d) % n, i] = 1
    return adj


class GNNAgent:
    """Graph Attention Network 기반 에이전트 (간소화)"""
    
    def __init__(self, svo_deg, agent_id, rng):
        self.svo = np.radians(svo_deg)
        self.lambda_t = np.sin(self.svo)
        self.id = agent_id
        self.rng = rng
        # Attention 가중치 (학습 시뮬레이션)
        self.w_svo = rng.normal(0.5, 0.1)
        self.w_lambda = rng.normal(0.3, 0.1)
        self.w_contrib = rng.normal(0.2, 0.1)
        self.attention_hist = []
    
    def compute_attention(self, neighbors_data):
        """이웃별 attention score 계산"""
        if not neighbors_data:
            return []
        
        scores = []
        for n_svo, n_lambda, n_contrib in neighbors_data:
            score = (self.w_svo * abs(np.sin(n_svo) - np.sin(self.svo)) +
                    self.w_lambda * abs(n_lambda - self.lambda_t) +
                    self.w_contrib * n_contrib / ENDOWMENT)
            scores.append(score)
        
        # Softmax
        scores = np.array(scores)
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / (np.sum(exp_scores) + 1e-10)
    
    def decide(self, resource, neighbors_data):
        attention = self.compute_attention(neighbors_data)
        self.attention_hist.append(attention.tolist() if len(attention) > 0 else [])
        
        if len(attention) > 0 and len(neighbors_data) > 0:
            neighbor_lambdas = [nd[1] for nd in neighbors_data]
            weighted_neighbor = np.sum(attention * neighbor_lambdas)
        else:
            weighted_neighbor = self.lambda_t
        
        base = np.sin(self.svo)
        if resource < 0.2:
            target = max(0, base * 0.3)
        elif resource > 0.7:
            target = min(1, base * 1.5)
        else:
            target = base * (0.5 + 0.5 * weighted_neighbor)
        
        self.lambda_t = 0.9 * self.lambda_t + 0.1 * target
        
        # 적응 학습 (가중치 업데이트)
        lr = 0.01
        self.w_svo += lr * self.rng.normal(0, 0.01)
        self.w_lambda += lr * self.rng.normal(0, 0.01)
        self.w_contrib += lr * self.rng.normal(0, 0.01)
        
        return int(np.clip(self.lambda_t * ENDOWMENT * 0.8 + self.rng.normal(0, 3), 0, ENDOWMENT))


class AvgAgent:
    """단순 평균 에이전트 (비교군)"""
    def __init__(self, svo_deg, rng):
        self.svo = np.radians(svo_deg)
        self.lambda_t = np.sin(self.svo)
        self.rng = rng
    
    def decide(self, resource, neighbors_data):
        if neighbors_data:
            avg_lambda = np.mean([nd[1] for nd in neighbors_data])
        else:
            avg_lambda = self.lambda_t
        
        base = np.sin(self.svo)
        if resource < 0.2:
            target = max(0, base * 0.3)
        elif resource > 0.7:
            target = min(1, base * 1.5)
        else:
            target = base * (0.5 + 0.5 * avg_lambda)
        
        self.lambda_t = 0.9 * self.lambda_t + 0.1 * target
        return int(np.clip(self.lambda_t * ENDOWMENT * 0.8 + self.rng.normal(0, 3), 0, ENDOWMENT))


def simulate_gnn(agent_type, svo_degs_mixed, seed):
    """GNN vs 평균 에이전트 시뮬레이션"""
    rng = np.random.RandomState(seed)
    adj = build_ring_graph(N_AGENTS, K)
    
    if agent_type == "gnn":
        agents = [GNNAgent(svo_degs_mixed[i], i, rng) for i in range(N_AGENTS)]
    else:
        agents = [AvgAgent(svo_degs_mixed[i], rng) for i in range(N_AGENTS)]
    
    resource = 0.5
    coop_hist, welfare_hist = [], []
    
    for t in range(N_STEPS):
        for i, agent in enumerate(agents):
            neighbors = np.where(adj[i] > 0)[0]
            neighbors_data = [(agents[j].svo, agents[j].lambda_t,
                             agents[j].lambda_t * ENDOWMENT * 0.8) for j in neighbors]
        
        contribs = []
        for i, agent in enumerate(agents):
            neighbors = np.where(adj[i] > 0)[0]
            nd = [(agents[j].svo, agents[j].lambda_t, agents[j].lambda_t * ENDOWMENT * 0.8) for j in neighbors]
            contribs.append(agent.decide(resource, nd))
        
        total = sum(contribs)
        public_good = total * 1.6 / N_AGENTS
        payoffs = [(ENDOWMENT - c) + public_good for c in contribs]
        
        resource = np.clip(resource + 0.02 * (np.mean(contribs) / ENDOWMENT - 0.3), 0, 1)
        coop_hist.append(float(np.mean([c > ENDOWMENT * 0.3 for c in contribs])))
        welfare_hist.append(float(np.mean(payoffs)))
    
    attention_data = None
    if agent_type == "gnn":
        attention_data = [a.attention_hist[-1] if a.attention_hist and a.attention_hist[-1] else [1/K]*K for a in agents]
    
    return {
        "mean_coop": float(np.mean(coop_hist[-30:])),
        "mean_welfare": float(np.mean(welfare_hist[-30:])),
        "coop_history": coop_hist,
        "welfare_history": welfare_hist,
        "attention_data": attention_data,
    }


def run_experiment():
    rng = np.random.RandomState(42)
    svo_mixed = [rng.choice([0, 15, 45, 90]) for _ in range(N_AGENTS)]
    
    results = {}
    for agent_type in ["gnn", "avg"]:
        runs = [simulate_gnn(agent_type, svo_mixed, s) for s in range(N_SEEDS)]
        results[agent_type] = {
            "mean_coop": float(np.mean([r["mean_coop"] for r in runs])),
            "mean_welfare": float(np.mean([r["mean_welfare"] for r in runs])),
            "coop_std": float(np.std([r["mean_coop"] for r in runs])),
            "avg_coop_hist": [float(np.mean([r["coop_history"][t] for r in runs])) for t in range(N_STEPS)],
            "avg_welfare_hist": [float(np.mean([r["welfare_history"][t] for r in runs])) for t in range(N_STEPS)],
        }
        if agent_type == "gnn":
            last_run = runs[-1]
            results["attention_sample"] = last_run["attention_data"]
    
    return results, svo_mixed


def plot_fig69(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Fig 69: GNN Agent vs Average Agent — Social Dilemma Performance",
                 fontsize=14, fontweight='bold', y=1.02)
    
    ax = axes[0]
    ax.plot(results["gnn"]["avg_coop_hist"], color='#1e88e5', linewidth=2, label='GNN Agent')
    ax.plot(results["avg"]["avg_coop_hist"], color='#e53935', linewidth=2, linestyle='--', label='Average Agent')
    ax.set_title('Cooperation Over Time', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Cooperation Rate')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(results["gnn"]["avg_welfare_hist"], color='#1e88e5', linewidth=2, label='GNN Agent')
    ax.plot(results["avg"]["avg_welfare_hist"], color='#e53935', linewidth=2, linestyle='--', label='Average Agent')
    ax.set_title('Social Welfare Over Time', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Welfare')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    labels = ['GNN Agent', 'Average Agent']
    coops = [results["gnn"]["mean_coop"], results["avg"]["mean_coop"]]
    welfares = [results["gnn"]["mean_welfare"], results["avg"]["mean_welfare"]]
    x = np.arange(2)
    ax.bar(x - 0.2, coops, 0.35, label='Cooperation', color='#1e88e5', alpha=0.8)
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, welfares, 0.35, label='Welfare', color='#43a047', alpha=0.8)
    ax.set_title('Final Performance Comparison', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Cooperation', color='#1e88e5'); ax2.set_ylabel('Welfare', color='#43a047')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig69_gnn_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[Q4] Fig 69 저장: {path}")


def plot_fig70(results, svo_mixed):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Fig 70: GNN Attention Analysis — Who Influences Whom?",
                 fontsize=14, fontweight='bold', y=1.02)
    
    attention_data = results.get("attention_sample")
    
    ax = axes[0]
    if attention_data:
        valid_attn = [a for a in attention_data if a and len(a) > 0]
        entropy = [float(-np.sum(np.array(a) * np.log(np.array(a) + 1e-10))) for a in valid_attn]
        max_entropy = -K * (1/K) * np.log(1/K)
        
        svo_labels = {'0': 'Selfish', '15': 'Indiv.', '45': 'Prosocial', '90': 'Altruist'}
        svo_groups = {}
        for i, s in enumerate(svo_mixed):
            key = str(int(s))
            if key not in svo_groups:
                svo_groups[key] = []
            if i < len(entropy):
                svo_groups[key].append(entropy[i])
        
        group_names = [svo_labels.get(k, k) for k in sorted(svo_groups.keys())]
        group_vals = [np.mean(svo_groups[k]) for k in sorted(svo_groups.keys())]
        colors = ['#e53935', '#ff9800', '#1e88e5', '#43a047']
        ax.bar(range(len(group_names)), group_vals, color=colors[:len(group_names)], alpha=0.8, edgecolor='black')
        ax.axhline(max_entropy, color='grey', linestyle='--', label=f'Uniform ({max_entropy:.2f})')
        ax.set_xticks(range(len(group_names))); ax.set_xticklabels(group_names)
        ax.set_title('Attention Entropy by SVO Group', fontweight='bold')
        ax.set_ylabel('Entropy (H)'); ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    
    ax = axes[1]
    if attention_data:
        sample_agent = 5
        if sample_agent < len(attention_data) and attention_data[sample_agent]:
            attn = attention_data[sample_agent]
            ax.bar(range(len(attn)), attn, color='#7e57c2', alpha=0.8, edgecolor='black')
            ax.set_title(f'Attention Weights (Agent #{sample_agent}, SVO={svo_mixed[sample_agent]}°)', fontweight='bold')
            ax.set_xlabel('Neighbor Index'); ax.set_ylabel('Attention Weight')
            ax.grid(True, axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No attention data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Attention Weights', fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig70_gnn_attention.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[Q4] Fig 70 저장: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  [Q4] GNN 에이전트")
    print("=" * 60)
    results, svo_mixed = run_experiment()
    print(f"\n  GNN:  Coop={results['gnn']['mean_coop']:.3f}, Welfare={results['gnn']['mean_welfare']:.1f}")
    print(f"  Avg:  Coop={results['avg']['mean_coop']:.3f}, Welfare={results['avg']['mean_welfare']:.1f}")
    print(f"  Δ Coop: {results['gnn']['mean_coop'] - results['avg']['mean_coop']:+.4f}")
    plot_fig69(results)
    plot_fig70(results, svo_mixed)
    json_path = os.path.join(OUTPUT_DIR, "gnn_results.json")
    with open(json_path, 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if 'hist' not in kk and 'sample' not in kk}
                  for k, v in results.items() if isinstance(v, dict)}, f, indent=2)
    print(f"\n[Q4] JSON: {json_path}")
