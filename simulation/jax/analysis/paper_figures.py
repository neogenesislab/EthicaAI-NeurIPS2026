"""
EthicaAI 논문 Figure/Table 생성기
Stage 2 sweep 데이터에서 추가 시각화 + LaTeX Table을 생성합니다.
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, Any

# 스타일 설정 (NeurIPS Publication-Ready)
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf',  # 기본 PDF (벡터)
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'text.usetex': False, # LaTeX 설치 필요시 True
})

# 색상 팔레트
COLORS = {
    "selfish": "#E74C3C",
    "individualist": "#F1C40F",
    "competitive": "#E67E22",
    "prosocial": "#2ECC71",
    "cooperative": "#95A5A6",
    "altruistic": "#3498DB",
    "full_altruist": "#9B59B6",
}


def _get_color(name):
    return COLORS.get(name, "#7F8C8D")


# ===========================================================
# Fig.6: 인과 분석 Forest Plot
# ===========================================================
def plot_causal_forest(causal_results: dict, output_dir: str):
    """
    H1/H2 ATE + 95% CI Forest Plot.
    인과 효과 크기와 신뢰구간을 한 눈에 보여줍니다.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    hypotheses = []
    ates = []
    lower = []
    upper = []
    colors_list = []

    for key in ["H1_svo_reward", "H2_svo_cooperation", "H3_svo_gini"]:
        r = causal_results.get(key, {})
        if "ate" not in r:
            continue
        ate = r["ate"]
        se = r["se"]
        ci_lo = ate - 1.96 * se
        ci_hi = ate + 1.96 * se
        sig = r.get("significant", False)

        label_map = {
            "H1_svo_reward": "H1: SVO → Reward",
            "H2_svo_cooperation": "H2: SVO → Cooperation",
            "H3_svo_gini": "H3: SVO → Gini",
        }
        hypotheses.append(label_map.get(key, key))
        ates.append(ate)
        lower.append(ci_lo)
        upper.append(ci_hi)
        colors_list.append("#2ECC71" if sig else "#E74C3C")

    y_pos = np.arange(len(hypotheses))

    # 수평 CI 바
    for i, (ate, lo, hi, color) in enumerate(zip(ates, lower, upper, colors_list)):
        ax.plot([lo, hi], [i, i], color=color, linewidth=3, solid_capstyle='round')
        ax.plot(ate, i, 'o', color=color, markersize=10, zorder=5)

    # 영가설 기준선
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(hypotheses)
    ax.set_xlabel("Average Treatment Effect (ATE)")
    ax.set_title("Causal Analysis: Forest Plot")
    ax.grid(True, alpha=0.2, axis='x')

    # p-value 주석
    for i, key in enumerate(["H1_svo_reward", "H2_svo_cooperation", "H3_svo_gini"]):
        r = causal_results.get(key, {})
        if i >= len(ates):
            continue

    path = os.path.join(output_dir, "fig6_causal_forest.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ===========================================================
# Fig.8: 종합 Heatmap (7 SVO x 4 메트릭)
# ===========================================================
def plot_summary_heatmap(sweep_results: dict, output_dir: str):
    """
    7 SVO 조건 x 4 메트릭 정규화 히트맵.
    모든 결과를 한 눈에 비교할 수 있습니다.
    """
    metric_keys = {
        "Reward": "reward_mean",
        "Cooperation": "cooperation_rate",
        "Gini": "gini",
        "theta_clean": "threshold_clean_mean",
    }

    svo_names = list(sweep_results.keys())
    data_matrix = np.zeros((len(svo_names), len(metric_keys)))

    for i, name in enumerate(svo_names):
        runs = sweep_results[name]["runs"]
        for j, (label, key) in enumerate(metric_keys.items()):
            values = [r["metrics"][key][-1] for r in runs]
            data_matrix[i, j] = np.mean(values)

    # 열(메트릭)별 min-max 정규화
    norm_matrix = np.zeros_like(data_matrix)
    for j in range(data_matrix.shape[1]):
        col = data_matrix[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 1e-10:
            norm_matrix[:, j] = (col - col_min) / (col_max - col_min)
        else:
            norm_matrix[:, j] = 0.5

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(norm_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # 축 레이블
    ax.set_xticks(np.arange(len(metric_keys)))
    ax.set_xticklabels(list(metric_keys.keys()))
    ax.set_yticks(np.arange(len(svo_names)))
    ax.set_yticklabels(svo_names)

    # 셀에 원본 값 표시
    for i in range(len(svo_names)):
        for j in range(len(metric_keys)):
            val = data_matrix[i, j]
            text_color = "white" if norm_matrix[i, j] < 0.3 or norm_matrix[i, j] > 0.7 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                   fontsize=9, color=text_color, fontweight='bold')

    ax.set_title("Summary Heatmap: SVO Conditions x Metrics (Normalized)")
    fig.colorbar(im, ax=ax, label="Normalized Value (0=min, 1=max)", shrink=0.8)

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    path = os.path.join(output_dir, "fig8_summary_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ===========================================================
# Table 1: 실험 설정 요약 (LaTeX)
# ===========================================================
def generate_config_table(config: dict, output_dir: str):
    """실험 설정 파라미터를 LaTeX 테이블로 생성."""
    rows = [
        ("환경", "Cleanup (SocialJax)"),
        ("에이전트 수 (N)", str(config.get("NUM_AGENTS", 20))),
        ("병렬 환경 (B)", str(config.get("NUM_ENVS", 16))),
        ("학습 에포크", str(config.get("NUM_UPDATES", 300))),
        ("롤아웃 길이", str(config.get("ROLLOUT_LEN", 128))),
        ("할인율 (gamma)", str(config.get("GAMMA", 0.99))),
        ("학습률", str(config.get("LR", 3e-4))),
        ("엔트로피 계수", str(config.get("ENTROPY_COEFF", 0.05))),
        ("메타랭킹 beta (psi)", str(config.get("META_BETA", 0.1))),
        ("생존 임계값", str(config.get("META_SURVIVAL_THRESHOLD", -5.0))),
        ("관대성 임계값", str(config.get("META_WEALTH_BOOST", 5.0))),
        ("SVO 조건", "7 (0 ~ 90 deg)"),
        ("시드 수", "5 (0, 42, 123, 256, 999)"),
        ("총 실행 수", "35"),
    ]

    # LaTeX 형식
    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\caption{Experimental Configuration}\n"
    latex += "\\label{tab:config}\n"
    latex += "\\begin{tabular}{ll}\n\\toprule\n"
    latex += "Parameter & Value \\\\\n\\midrule\n"
    for param, value in rows:
        latex += f"{param} & {value} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    path = os.path.join(output_dir, "table1_config.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  Saved: {path}")

    # Markdown 형식도 생성
    md = "| Parameter | Value |\n|-----------|-------|\n"
    for param, value in rows:
        md += f"| {param} | {value} |\n"

    md_path = os.path.join(output_dir, "table1_config.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Saved: {md_path}")

    return path


# ===========================================================
# Table 2: Stage 2 결과 요약 (LaTeX)
# ===========================================================
def generate_results_table(sweep_results: dict, output_dir: str):
    """7 SVO 조건별 결과 요약 테이블 생성."""
    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\caption{Stage 2 Results Summary (7 SVO x 5 Seeds)}\n"
    latex += "\\label{tab:results}\n"
    latex += "\\begin{tabular}{lcccccc}\n\\toprule\n"
    latex += "SVO & $\\theta$ (rad) & Reward & Coop\\% & Gini & $\\theta_{clean}$ & $\\theta_{harv}$ \\\\\n"
    latex += "\\midrule\n"

    md = "| SVO | theta | Reward | Coop% | Gini | theta_clean | theta_harv |\n"
    md += "|-----|-------|--------|-------|------|-------------|------------|\n"

    for name, data in sweep_results.items():
        theta = data["theta"]
        runs = data["runs"]

        rewards = [r["metrics"]["reward_mean"][-1] for r in runs]
        coops = [r["metrics"]["cooperation_rate"][-1] for r in runs]
        ginis = [r["metrics"]["gini"][-1] for r in runs]
        cleans = [r["metrics"]["threshold_clean_mean"][-1] for r in runs]
        harvs = [r["metrics"]["threshold_harvest_mean"][-1] for r in runs]

        r_mean, r_se = np.mean(rewards), np.std(rewards) / np.sqrt(len(rewards))
        c_mean = np.mean(coops)
        g_mean, g_se = np.mean(ginis), np.std(ginis) / np.sqrt(len(ginis))
        cl_mean = np.mean(cleans)
        hv_mean = np.mean(harvs)

        latex += f"{name} & {theta:.4f} & {r_mean:.4f}$\\pm${r_se:.4f} & "
        latex += f"{c_mean:.4f} & {g_mean:.4f}$\\pm${g_se:.4f} & "
        latex += f"{cl_mean:.4f} & {hv_mean:.4f} \\\\\n"

        md += f"| {name} | {theta:.4f} | {r_mean:.4f}+/-{r_se:.4f} | "
        md += f"{c_mean:.4f} | {g_mean:.4f}+/-{g_se:.4f} | "
        md += f"{cl_mean:.4f} | {hv_mean:.4f} |\n"

    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    path = os.path.join(output_dir, "table2_results.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  Saved: {path}")

    md_path = os.path.join(output_dir, "table2_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Saved: {md_path}")

    return path


# ===========================================================
# Table 3: 인과 분석 결과 (LaTeX)
# ===========================================================
def generate_causal_table(causal_results: dict, output_dir: str):
    """인과 분석 결과 테이블 생성."""
    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\caption{Causal Analysis Results}\n"
    latex += "\\label{tab:causal}\n"
    latex += "\\begin{tabular}{lccccc}\n\\toprule\n"
    latex += "Hypothesis & ATE & SE & t & p-value & Sig. \\\\\n"
    latex += "\\midrule\n"

    md = "| Hypothesis | ATE | SE | t | p-value | f^2 | Sig. |\n"
    md += "|------------|-----|-----|---|---------|-----|------|\n"

    for key in ["H1_svo_reward", "H2_svo_cooperation", "H3_svo_gini"]:
        r = causal_results.get(key, {})
        if "ate" not in r:
            continue

        label_map = {
            "H1_svo_reward": "H1: SVO $\\to$ Reward",
            "H2_svo_cooperation": "H2: SVO $\\to$ Cooperation",
            "H3_svo_gini": "H3: SVO $\\to$ Gini",
        }
        label_md_map = {
            "H1_svo_reward": "H1: SVO -> Reward",
            "H2_svo_cooperation": "H2: SVO -> Cooperation",
            "H3_svo_gini": "H3: SVO -> Gini",
        }

        ate, se, t, p = r["ate"], r["se"], r["t_stat"], r["p_value"]
        sig = "Yes" if r.get("significant") else "No"
        f2 = r.get("cohens_f2", 0)
        f2_str = f"{f2:.3f}" if f2 else "-"
        effect = r.get("effect_label", "")

        p_str = "<0.0001" if p < 0.0001 else f"{p:.4f}"
        sig_mark = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))

        latex += f"{label_map.get(key, key)} & {ate:.4f} & {se:.4f} & {t:.2f} & {p_str} & {sig}{sig_mark} \\\\\n"
        md += f"| {label_md_map.get(key, key)} | {ate:.4f} | {se:.4f} | {t:.2f} | {p_str} | {f2_str} ({effect}) | {sig}{sig_mark} |\n"

    # H1b
    h1b = causal_results.get("H1b_quadratic", {})
    if h1b:
        bl, bq = h1b.get("beta_linear", 0), h1b.get("beta_quadratic", 0)
        inv_u = h1b.get("inverted_u", "False")
        latex += f"H1b: Quadratic & $\\beta_l$={bl:.4f} & $\\beta_q$={bq:.4f} & - & - & Inv-U: {inv_u} \\\\\n"
        md += f"| H1b: Quadratic | bl={bl:.4f} | bq={bq:.4f} | - | - | - | Inv-U: {inv_u} |\n"

    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    path = os.path.join(output_dir, "table3_causal.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  Saved: {path}")

    md_path = os.path.join(output_dir, "table3_causal.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Saved: {md_path}")

    return path


# ===========================================================
# Main: Stage 2 데이터에서 모든 추가 Figure + Table 생성
# ===========================================================
def main():
    """Stage 2 sweep 결과에서 추가 논문 Figure/Table을 생성합니다."""
    if len(sys.argv) < 2:
        print("Usage: python paper_figures.py <run_dir>")
        print("Example: python paper_figures.py simulation/outputs/run_medium_1770985029")
        sys.exit(1)

    run_dir = sys.argv[1]

    # sweep JSON 로드
    sweep_files = [f for f in os.listdir(run_dir) if f.startswith("sweep_") and f.endswith(".json")]
    if not sweep_files:
        print(f"No sweep JSON found in {run_dir}")
        sys.exit(1)

    sweep_path = os.path.join(run_dir, sweep_files[0])
    print(f"Loading sweep data: {sweep_path}")
    with open(sweep_path, "r", encoding="utf-8") as f:
        sweep_results = json.load(f)

    # 인과분석 JSON 로드
    causal_path = os.path.join(run_dir, "causal_results.json")
    print(f"Loading causal data: {causal_path}")
    with open(causal_path, "r", encoding="utf-8") as f:
        causal_results = json.load(f)

    # 출력 디렉터리
    fig_dir = os.path.join(run_dir, "figures")
    table_dir = os.path.join(run_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    print("\n=== 추가 Figure 생성 ===")
    plot_causal_forest(causal_results, fig_dir)
    plot_summary_heatmap(sweep_results, fig_dir)
    plot_lambda_mechanism(fig_dir)
    plot_role_specialization(sweep_results, fig_dir)  # Added Fig.9
    convergence_analysis(sweep_results, fig_dir)

    print("\n=== Table 생성 ===")
    # config 로드 (기본값 사용)
    config = {
        "NUM_AGENTS": 20, "NUM_ENVS": 16, "NUM_UPDATES": 300,
        "ROLLOUT_LEN": 128, "GAMMA": 0.99, "LR": 3e-4,
        "ENTROPY_COEFF": 0.05, "META_BETA": 0.1,
        "META_SURVIVAL_THRESHOLD": -5.0, "META_WEALTH_BOOST": 5.0,
    }
    generate_config_table(config, table_dir)
    generate_results_table(sweep_results, table_dir)
    generate_causal_table(causal_results, table_dir)
    generate_monotonicity_table(causal_results, table_dir)

    print(f"\nDone! Output: {run_dir}")


# ===========================================================
# Fig.4: Meta-Ranking 메커니즘 도식 (이론적 Lambda 궤적)
# ===========================================================
def plot_lambda_mechanism(output_dir: str):
    """Fig.4: 메타랭킹의 동적 헌신(lambda) 메커니즘 이론 시각화."""
    theta_range = np.linspace(0, np.pi/2, 100)
    lambda_base = np.sin(theta_range)
    lambda_generous = np.minimum(1.0, lambda_base * 1.5)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 좌: SVO별 Lambda 영역
    ax1 = axes[0]
    ax1.fill_between(theta_range, 0, lambda_base * 0.3, alpha=0.2, color='#E74C3C', label='Survival (λ→0)')
    ax1.fill_between(theta_range, lambda_base * 0.3, lambda_base, alpha=0.2, color='#3498DB', label='Normal (λ=sin θ)')
    ax1.fill_between(theta_range, lambda_base, lambda_generous, alpha=0.2, color='#2ECC71', label='Generosity (λ=1.5·sin θ)')
    ax1.plot(theta_range, lambda_base, 'b-', linewidth=2, label='λ_base')
    ax1.plot(theta_range, lambda_generous, 'g--', linewidth=2, label='λ_generous')
    
    # 7 SVO 포인트 표시
    svo_thetas = [0, 0.2618, 0.5236, 0.7854, 1.0472, 1.3090, 1.5708]
    svo_labels = ['SEL', 'IND', 'CMP', 'PRO', 'COP', 'ALT', 'F.A']
    svo_lambdas = [np.sin(t) for t in svo_thetas]
    ax1.scatter(svo_thetas, svo_lambdas, c=[_get_color(n) for n in COLORS.keys()], 
               s=80, zorder=5, edgecolors='black', linewidths=1)
    for th, lb, lam in zip(svo_thetas, svo_labels, svo_lambdas):
        ax1.annotate(lb, (th, lam), textcoords='offset points', xytext=(5, 8), fontsize=8)
    
    ax1.set_xlabel('SVO Angle (θ, radians)')
    ax1.set_ylabel('Commitment Level (λ)')
    ax1.set_title('(a) Dynamic Commitment Regions')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlim(0, np.pi/2)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.2)
    
    # 우: 보상 변환 수식 시각화
    ax2 = axes[1]
    lambdas = np.linspace(0, 1, 100)
    u_self = -0.15  # 대표값
    u_meta = -0.10  # 대표값
    psi = 0.005     # 대표값
    
    r_total = (1 - lambdas) * u_self + lambdas * (u_meta - psi)
    r_baseline = u_self * np.cos(np.arcsin(lambdas)) + u_meta * lambdas
    
    ax2.plot(lambdas, r_total, 'b-', linewidth=2, label='Meta-Ranking')
    ax2.plot(lambdas, r_baseline, 'r--', linewidth=2, label='Baseline (SVO only)')
    ax2.fill_between(lambdas, r_baseline, r_total, alpha=0.15, color='blue',
                     where=r_total > r_baseline, label='Meta-Ranking Gain')
    ax2.set_xlabel('Commitment Level (λ)')
    ax2.set_ylabel('Total Reward (R_total)')
    ax2.set_title('(b) Reward Transformation')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    
    fig.suptitle('Fig.4: Meta-Ranking Mechanism', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(output_dir, "fig4_mechanism.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ===========================================================
# Fig.9: 역할 분화 (Role Specialization) Scatter Plot
# ===========================================================
def plot_role_specialization(sweep_results: dict, output_dir: str):
    """
    Fig.9: Cleaner vs Eater 역할 분화 시각화.
    X축: Harvest Propensity (섭취 성향, -Threshold)
    Y축: Cleaning Propensity (청소 성향, -Threshold)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    conditions = ["selfish", "individualist", "competitive", "prosocial", "cooperative", "altruistic", "full_altruist"]
    markers = {
        "selfish": "o", "individualist": "v", "competitive": "^",
        "prosocial": "s", "cooperative": "P", "altruistic": "*", "full_altruist": "D"
    }

    for cond_name in conditions:
        if cond_name not in sweep_results:
            continue
            
        data = sweep_results[cond_name]
        runs = data["runs"]
        
        all_cleans = []
        all_eats = []
        
        for run in runs:
            # final_thresholds: shape (N, 2)
            if "final_thresholds" in run:
                th = np.array(run["final_thresholds"])
                # Threshold가 낮을수록 해당 행동을 많이 함 -> 역전
                clean_propensity = -th[:, 0]
                eat_propensity = -th[:, 1]
                
                all_cleans.extend(clean_propensity)
                all_eats.extend(eat_propensity)

        if all_cleans:
            ax.scatter(all_eats, all_cleans, label=cond_name, 
                       c=_get_color(cond_name), marker=markers.get(cond_name, "o"), 
                       alpha=0.6, edgecolors='w', s=60)

    ax.set_xlabel("Harvest Propensity (Higher = More Greedy)")
    ax.set_ylabel("Cleaning Propensity (Higher = More Altruistic)")
    ax.set_title("Emergent Role Specialization by SVO")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "fig9_role_specialization.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ===========================================================
# B4: 수렴 분석 (변동계수 CV)
# ===========================================================
def convergence_analysis(sweep_results: dict, output_dir: str, window: int = 50):
    """마지막 window 에포크의 변동계수(CV) 계산 + Figure 생성."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    cvs = []
    names = []
    
    for name, data in sweep_results.items():
        run_cvs = []
        for run in data["runs"]:
            tail = run["metrics"]["reward_mean"][-window:]
            mean_val = np.mean(tail)
            std_val = np.std(tail)
            cv = std_val / (np.abs(mean_val) + 1e-10)
            run_cvs.append(cv)
        cvs.append(run_cvs)
        names.append(name)
    
    # Box plot
    bp = ax.boxplot(cvs, labels=names, patch_artist=True, widths=0.6)
    for i, (patch, name) in enumerate(zip(bp['boxes'], names)):
        patch.set_facecolor(_get_color(name))
        patch.set_alpha(0.6)
    
    # 수렴 기준선 (CV < 0.05)
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Convergence Threshold (CV=0.05)')
    
    ax.set_xlabel('SVO Condition')
    ax.set_ylabel(f'Coefficient of Variation (last {window} epochs)')
    ax.set_title('Convergence Analysis')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    path = os.path.join(output_dir, "fig_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ===========================================================
# Table 4: 단조성 검정 결과
# ===========================================================
def generate_monotonicity_table(causal_results: dict, output_dir: str):
    """단조성 검정 결과(Spearman + Kruskal-Wallis)를 테이블로 생성."""
    mono = causal_results.get("monotonicity", {})
    if not mono:
        print("  (monotonicity data not found -- skip)")
        return None
    
    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\caption{Monotonicity and Group Difference Tests}\n"
    latex += "\\label{tab:monotonicity}\n"
    latex += "\\begin{tabular}{lcccc}\n\\toprule\n"
    latex += "Test & Metric & Statistic & p-value & Sig. \\\\\n"
    latex += "\\midrule\n"
    
    md = "| Test | Metric | Statistic | p-value | Sig. |\n"
    md += "|------|--------|-----------|---------|------|\n"
    
    for key, val in mono.items():
        if "spearman" in key:
            metric = key.replace("spearman_", "").capitalize()
            rho = val["rho"]
            p = val["p"]
            sig = "Yes***" if p < 0.001 else ("Yes**" if p < 0.01 else ("Yes*" if p < 0.05 else "No"))
            p_str = "<0.0001" if p < 0.0001 else f"{p:.4f}"
            latex += f"Spearman & {metric} & rho={rho:.4f} & {p_str} & {sig} \\\\\n"
            md += f"| Spearman | {metric} | rho={rho:.4f} | {p_str} | {sig} |\n"
        elif "kruskal" in key:
            metric = key.replace("kruskal_", "").capitalize()
            H = val["H"]
            p = val["p"]
            sig = "Yes***" if p < 0.001 else ("Yes**" if p < 0.01 else ("Yes*" if p < 0.05 else "No"))
            p_str = "<0.0001" if p < 0.0001 else f"{p:.4f}"
            latex += f"Kruskal-Wallis & {metric} & H={H:.2f} & {p_str} & {sig} \\\\\n"
            md += f"| Kruskal-Wallis | {metric} | H={H:.2f} | {p_str} | {sig} |\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    
    path = os.path.join(output_dir, "table4_monotonicity.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  Saved: {path}")
    
    md_path = os.path.join(output_dir, "table4_monotonicity.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Saved: {md_path}")
    
    return path


if __name__ == "__main__":
    main()
