"""
EthicaAI Full Pipeline Runner
Sweep → Evaluate → Causal → Visualize → Report
"""
import json
import os
import time

from simulation.jax.experiment_jax import run_sweep, print_summary_table, SVO_ANGLES
from simulation.jax.analysis.eval_jax import evaluate_sweep
from simulation.jax.analysis.causal_jax import run_causal_analysis, print_causal_report
from simulation.jax.analysis.visualize import generate_all_figures


def run_full_pipeline(scale="small", svo_angles=None, seeds=None,
                       output_dir="simulation/outputs", config_override=None):
    """
    End-to-End 논문 데이터 생산 파이프라인.
    
    1. SVO Sweep 실험 실행
    2. 평가 메트릭 집계
    3. 인과추론 분석
    4. 논문 Figure 생성
    5. 결과 요약 보고서 저장
    
    Args:
        config_override: config dict에 덮어쓸 키-값 쌍 (예: {"USE_META_RANKING": False})
    """
    if svo_angles is None:
        svo_angles = SVO_ANGLES
    if seeds is None:
        seeds = [0, 42, 123]
    
    timestamp = int(time.time())
    run_dir = os.path.join(output_dir, f"run_{scale}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # config_override 저장 (재현성)
    if config_override:
        override_path = os.path.join(run_dir, "config_override.json")
        with open(override_path, "w", encoding="utf-8") as f:
            json.dump(config_override, f, indent=2)
        print(f"  Config Override: {config_override}")
    
    print("=" * 60)
    print("EthicaAI Full Pipeline")
    print(f"Output: {run_dir}")
    print("=" * 60)
    
    # Step 1: SVO Sweep (config_override 전달)
    print("\n[1/4] Running SVO Sweep...")
    sweep_results, sweep_path = run_sweep(
        scale=scale, svo_angles=svo_angles, seeds=seeds,
        output_dir=run_dir, config_override=config_override
    )
    print_summary_table(sweep_results)
    
    # Step 2: Evaluate
    print("\n[2/4] Running Evaluation...")
    eval_results = evaluate_sweep(sweep_results)
    
    eval_path = os.path.join(run_dir, "eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"  Saved: {eval_path}")
    
    # Step 3: Causal Analysis
    print("\n[3/4] Running Causal Analysis...")
    causal_results = run_causal_analysis(sweep_results)
    print_causal_report(causal_results)
    
    causal_path = os.path.join(run_dir, "causal_results.json")
    with open(causal_path, "w", encoding="utf-8") as f:
        json.dump(causal_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"  Saved: {causal_path}")
    
    # Step 4: Generate Figures
    print("\n[4/4] Generating Figures...")
    fig_dir = os.path.join(run_dir, "figures")
    figure_paths = generate_all_figures(sweep_results, fig_dir)
    
    # Summary Report
    summary = {
        "timestamp": timestamp,
        "scale": scale,
        "config_override": config_override or {},
        "svo_conditions": len(svo_angles),
        "seeds": seeds,
        "total_runs": len(svo_angles) * len(seeds),
        "eval_summary": {
            name: {
                "reward": data.get("reward", {}),
                "cooperation": data.get("cooperation", {}),
                "gini": data.get("gini", {}),
            }
            for name, data in eval_results.items()
            if name != "comparison"
        },
        "causal_summary": {
            name: {
                "ate": r.get("ate"),
                "p_value": r.get("p_value"),
                "significant": r.get("significant"),
                "cohens_f2": r.get("cohens_f2"),
                "effect_label": r.get("effect_label"),
            }
            for name, r in causal_results.items()
            if "ate" in r
        },
        "monotonicity": causal_results.get("monotonicity", {}),
        "figures": figure_paths,
    }
    
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Pipeline Complete!")
    print(f"Results: {run_dir}")
    print(f"{'='*60}")
    
    return summary, run_dir


# ============================================================
# EXECUTION MODES
# ============================================================

# --- Stage 1: Pilot (수렴 확인용) ---
PILOT_ANGLES = {
    "selfish": SVO_ANGLES["selfish"],
    "prosocial": SVO_ANGLES["prosocial"],
    "full_altruist": SVO_ANGLES["full_altruist"],
}
PILOT_SEEDS = [0, 42, 123]

# --- Stage 2: Full Experiment (논문 데이터) ---
FULL_SEEDS = [0, 42, 123, 256, 999]
FULL_SEEDS_10 = [0, 42, 123, 256, 999, 1337, 2024, 3141, 4269, 5555]

# --- Stage 3: Large Scale (견고성) ---
LARGE_ANGLES = {
    "selfish": SVO_ANGLES["selfish"],
    "prosocial": SVO_ANGLES["prosocial"],
    "full_altruist": SVO_ANGLES["full_altruist"],
}
LARGE_SEEDS = [0, 42, 123, 256, 999]

# --- Stage 3+: Large Scale Full (NeurIPS 2026 보강) ---
LARGE_FULL_ANGLES = SVO_ANGLES  # 7개 SVO 전체
LARGE_FULL_SEEDS = FULL_SEEDS_10  # 10개 시드

# --- Ablation: 핵심 파라미터 비활성화 ---
ABLATION_ANGLES = {
    "selfish": SVO_ANGLES["selfish"],
    "prosocial": SVO_ANGLES["prosocial"],
    "full_altruist": SVO_ANGLES["full_altruist"],
}
ABLATION_SEEDS = [0, 42, 123]


if __name__ == "__main__":
    import sys
    
    stage = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    
    if stage == "pilot":
        print(">>> STAGE 1: PILOT (Medium, 3 SVO x 3 seeds)")
        summary, path = run_full_pipeline(
            scale="medium",
            svo_angles=PILOT_ANGLES,
            seeds=PILOT_SEEDS,
        )
    elif stage == "full":
        print(">>> STAGE 2: FULL EXPERIMENT (Medium, 7 SVO x 5 seeds)")
        summary, path = run_full_pipeline(
            scale="medium",
            svo_angles=SVO_ANGLES,
            seeds=FULL_SEEDS,
        )
    elif stage == "large":
        print(">>> STAGE 3: LARGE SCALE (Large, 3 SVO x 5 seeds)")
        summary, path = run_full_pipeline(
            scale="large",
            svo_angles=LARGE_ANGLES,
            seeds=LARGE_SEEDS,
        )
    elif stage == "baseline":
        print(">>> BASELINE: Meta-Ranking OFF (Medium, 7 SVO x 5 seeds)")
        summary, path = run_full_pipeline(
            scale="medium",
            svo_angles=SVO_ANGLES,
            seeds=FULL_SEEDS,
            config_override={"USE_META_RANKING": False},
        )
    elif stage == "ablation_nopsi":
        print(">>> ABLATION: No-Psi (META_BETA=0, 3 SVO x 3 seeds)")
        summary, path = run_full_pipeline(
            scale="medium",
            svo_angles=ABLATION_ANGLES,
            seeds=ABLATION_SEEDS,
            config_override={"META_BETA": 0.0},
        )
    elif stage == "ablation_static":
        print(">>> ABLATION: Static-Lambda (3 SVO x 3 seeds)")
        summary, path = run_full_pipeline(
            scale="medium",
            svo_angles=ABLATION_ANGLES,
            seeds=ABLATION_SEEDS,
            config_override={"META_USE_DYNAMIC_LAMBDA": False},
        )
    # === NeurIPS 2026 보강 실험 ===
    elif stage == "large_full":
        print(">>> STAGE 3+: LARGE FULL (Large, 7 SVO x 10 seeds = 70 runs)")
        summary, path = run_full_pipeline(
            scale="large",
            svo_angles=LARGE_FULL_ANGLES,
            seeds=LARGE_FULL_SEEDS,
        )
    elif stage == "large_baseline":
        print(">>> LARGE BASELINE: Meta-Ranking OFF (Large, 7 SVO x 10 seeds)")
        summary, path = run_full_pipeline(
            scale="large",
            svo_angles=LARGE_FULL_ANGLES,
            seeds=LARGE_FULL_SEEDS,
            config_override={"USE_META_RANKING": False},
        )
    elif stage == "large_harvest":
        print(">>> LARGE HARVEST: Cleanup→Harvest (Large, 7 SVO x 10 seeds)")
        summary, path = run_full_pipeline(
            scale="large_harvest",
            svo_angles=LARGE_FULL_ANGLES,
            seeds=LARGE_FULL_SEEDS,
        )
    else:
        print(f"Unknown stage: {stage}")
        print("Usage: python run_full_pipeline.py [pilot|full|large|baseline|ablation_nopsi|ablation_static|large_full|large_baseline|large_harvest]")

