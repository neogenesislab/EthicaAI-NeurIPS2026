import os
import sys
import time

# Add current directory to path so that 'simulation' module can be found
sys.path.append(os.getcwd())

from simulation.jax.experiment_jax import run_sweep
from simulation.jax.config import SVO_SWEEP_THETAS

def run_genesis_experiment():
    print(">>> Starting Genesis Run #001 (Hypothesis Verification) <<<")
    # 검증을 위해 'Full Altruist' 조건에서 테스트 (가장 불안정한 케이스)
    genesis_angles = {
        "genesis_test": SVO_SWEEP_THETAS["full_altruist"]
    }
    
    # 빠른 검증을 위해 Small Scale로 1개 시드만 실행
    output_dir = "simulation/outputs/genesis_001"
    os.makedirs(output_dir, exist_ok=True)
    
    results, path = run_sweep(
        scale="small",
        svo_angles=genesis_angles,
        seeds=[1004], # Angel Seed
        output_dir=output_dir
    )
    
    print(f"Genesis Run Complete. Results at {path}")

if __name__ == "__main__":
    run_genesis_experiment()
