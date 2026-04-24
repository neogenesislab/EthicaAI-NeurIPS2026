"""
O-I1: Lyapunov 안정성 분석
EthicaAI Phase O — 연구축 I: 이론 심화

동적 λ_t의 수렴성을 Lyapunov 안정성 이론으로 분석합니다.
- V(λ) = (λ - λ*)^2 를 Lyapunov 후보 함수로 설정
- dV/dt < 0 임을 다양한 초기조건에서 검증
- 고정점(Fixed Point) 존재성을 시뮬레이션으로 입증

출력: 수렴 분석 결과 JSON + 시각화 (논문 Theorem 후보)
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


def lambda_dynamics(lambda_t, resource, svo_theta_deg):
    """λ_t의 동역학 — 자원 수준에 따른 업데이트 규칙

    dλ/dt = f(λ, R, θ)
    
    이 함수는 메타랭킹의 핵심 업데이트 규칙입니다:
    - 자원 부족(R<0.2): λ → 0 (생존 모드)
    - 자원 풍부(R>0.7): λ → min(1, λ_base * 1.5) (관대 모드)
    - 중간: λ → λ_base (기본 모드)
    """
    svo_theta = np.radians(svo_theta_deg)
    lambda_base = np.sin(svo_theta)
    
    # 목표점(attractor)
    if resource < 0.2:
        lambda_target = 0.0
    elif resource > 0.7:
        lambda_target = min(1.0, lambda_base * 1.5)
    else:
        lambda_target = lambda_base
    
    # 수렴 속도 (조절 파라미터)
    alpha = 0.1  # 학습률
    
    # 업데이트 방향
    delta = alpha * (lambda_target - lambda_t)
    
    return delta, lambda_target


def lyapunov_function(lambda_t, lambda_star):
    """Lyapunov 후보 함수: V(λ) = (λ - λ*)^2"""
    return (lambda_t - lambda_star) ** 2


def verify_lyapunov_stability(svo_theta_deg, n_initial=50, n_steps=500):
    """Lyapunov 안정성 검증

    Theorem (추측): 임의의 초기 λ_0 ∈ [0,1]에 대해,
    자원 수준 R이 일정 구간에 고정되면 λ_t → λ* 수렴.
    
    증명 전략:
    1. V(λ) = (λ - λ*)^2 >= 0 (양정치)
    2. dV/dt = 2(λ - λ*) · dλ/dt < 0 (감소 조건)
    3. dλ/dt = α(λ* - λ) 이므로 dV/dt = -2α(λ - λ*)^2 < 0
    """
    results = {}
    
    for resource_regime, resource_val in [("scarce", 0.1), ("normal", 0.5), ("abundant", 0.8)]:
        trajectories = []
        v_decreasing_count = 0
        convergence_steps = []
        
        for i in range(n_initial):
            lambda_0 = i / (n_initial - 1)  # 균등 분포 초기값
            lambda_t = lambda_0
            
            trajectory = [lambda_t]
            v_values = []
            
            converged_step = n_steps  # 기본: 미수렴
            
            for step in range(n_steps):
                delta, lambda_star = lambda_dynamics(lambda_t, resource_val, svo_theta_deg)
                lambda_t = np.clip(lambda_t + delta, 0, 1)
                trajectory.append(lambda_t)
                
                v = lyapunov_function(lambda_t, lambda_star)
                v_values.append(v)
                
                # 수렴 판정 (|λ - λ*| < ε)
                if abs(lambda_t - lambda_star) < 0.001 and converged_step == n_steps:
                    converged_step = step
            
            # V 감소 검증
            if len(v_values) > 1:
                v_diffs = np.diff(v_values)
                if np.all(v_diffs[v_diffs != 0] < 0):  # 모든 비-영 변화가 감소
                    v_decreasing_count += 1
            
            trajectories.append(trajectory)
            convergence_steps.append(converged_step)
        
        # 마지막 열린 구간의 λ*
        _, lambda_star_final = lambda_dynamics(0.5, resource_val, svo_theta_deg)
        
        results[resource_regime] = {
            "resource": resource_val,
            "lambda_star": float(lambda_star_final),
            "convergence_rate": float(v_decreasing_count / n_initial * 100),
            "mean_convergence_steps": float(np.mean([s for s in convergence_steps if s < n_steps])) if any(s < n_steps for s in convergence_steps) else float('inf'),
            "all_converged": bool(all(s < n_steps for s in convergence_steps)),
            "trajectories": [t[:100] for t in trajectories[:10]],  # 샘플만 저장
        }
    
    return results


def run_full_analysis():
    """전체 SVO × 자원 체제 분석"""
    all_results = {}
    
    svo_conditions = {"selfish": 0.0, "prosocial": 45.0, "altruistic": 90.0}
    
    for svo_name, svo_theta in svo_conditions.items():
        all_results[svo_name] = verify_lyapunov_stability(svo_theta)
    
    return all_results


def plot_convergence(results):
    """수렴 궤적 시각화"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Lyapunov Stability Analysis: λ_t Convergence to Fixed Points",
                 fontsize=14, fontweight='bold', y=0.98)
    
    svo_names = ["selfish", "prosocial", "altruistic"]
    regimes = ["scarce", "normal", "abundant"]
    colors_traj = plt.cm.viridis(np.linspace(0.2, 0.9, 10))
    
    for i, svo_name in enumerate(svo_names):
        for j, regime in enumerate(regimes):
            ax = axes[i, j]
            data = results[svo_name][regime]
            
            # 궤적 플롯
            for k, traj in enumerate(data["trajectories"]):
                ax.plot(traj, color=colors_traj[k], linewidth=0.8, alpha=0.7)
            
            # 고정점 표시
            ax.axhline(data["lambda_star"], color='red', linestyle='--', linewidth=2,
                       label=f'λ*={data["lambda_star"]:.2f}')
            
            ax.set_title(f'{svo_name.capitalize()} / {regime.capitalize()}', fontsize=10,
                        fontweight='bold')
            ax.set_xlabel('Step' if i == 2 else '')
            ax.set_ylabel('λ_t' if j == 0 else '')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            
            # 수렴률 텍스트
            ax.text(0.98, 0.02, f'Conv: {data["convergence_rate"]:.0f}%',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, "lyapunov_convergence.png")
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[O-I1] 수렴 분석 시각화: {path}")
    return path


def generate_theorem_draft(results):
    """Theorem 초안 생성"""
    theorem = """
=== Theorem 1 (λ_t Convergence — 추측) ===

Let λ_t ∈ [0,1] be the dynamic commitment level of an agent with 
SVO parameter θ ∈ [0°, 90°] and resource level R_t ∈ ℝ+.

Define the update rule:
    λ_{t+1} = λ_t + α(λ*(R_t, θ) - λ_t)

where λ*(R, θ) is the target:
    λ*(R, θ) = { 0                          if R < τ_survival
               { sin(θ)                     if τ_survival ≤ R ≤ τ_abundant
               { min(1, 1.5·sin(θ))         if R > τ_abundant

And V(λ) = (λ - λ*)² is the Lyapunov candidate function.

Then for any fixed resource regime (R = const):
    (a) V(λ) ≥ 0 for all λ (positive definite)
    (b) ΔV = V(λ_{t+1}) - V(λ_t) = -2α(1-α)(λ_t - λ*)² < 0 for λ_t ≠ λ*
    (c) Therefore λ_t → λ* as t → ∞ (asymptotic stability)

Proof sketch:
    ΔV = (λ_{t+1} - λ*)² - (λ_t - λ*)²
        = ((1-α)λ_t + αλ* - λ*)² - (λ_t - λ*)²
        = (1-α)²(λ_t - λ*)² - (λ_t - λ*)²
        = [(1-α)² - 1](λ_t - λ*)²
        = -α(2-α)(λ_t - λ*)²  < 0  for 0 < α < 2  ∎

Convergence rate: geometric with ratio (1-α)²
    For α = 0.1: |λ_t - λ*| ≤ (0.9)^t · |λ_0 - λ*| → 0

=== Empirical Verification ===
"""
    
    for svo_name, svo_results in results.items():
        theorem += f"\n{svo_name}:\n"
        for regime, data in svo_results.items():
            theorem += f"  {regime}: λ*={data['lambda_star']:.3f}, Conv={data['convergence_rate']:.0f}%, "
            theorem += f"Steps={data['mean_convergence_steps']:.0f}\n"
    
    return theorem


if __name__ == "__main__":
    print("=" * 60)
    print("  [O-I1] Lyapunov 안정성 분석")
    print("=" * 60)
    
    results = run_full_analysis()
    
    print("\n--- 수렴 분석 결과 ---")
    for svo_name in results:
        print(f"\n  [{svo_name}]")
        for regime, data in results[svo_name].items():
            print(f"    {regime:>8s}: λ*={data['lambda_star']:.3f} | Conv={data['convergence_rate']:.0f}% | "
                  f"Steps={data['mean_convergence_steps']:.0f} | All={data['all_converged']}")
    
    plot_convergence(results)
    
    theorem = generate_theorem_draft(results)
    theorem_path = os.path.join(OUTPUT_DIR, "theorem_draft.txt")
    with open(theorem_path, 'w', encoding='utf-8') as f:
        f.write(theorem)
    print(f"\n[O-I1] Theorem 초안: {theorem_path}")
    
    # JSON 저장
    json_data = {sn: {rg: {k: v for k, v in rd.items() if k != "trajectories"}
                       for rg, rd in sr.items()} for sn, sr in results.items()}
    json_path = os.path.join(OUTPUT_DIR, "lyapunov_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"[O-I1] 결과 JSON: {json_path}")
