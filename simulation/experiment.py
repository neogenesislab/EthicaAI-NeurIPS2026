"""
EthicaAI 실험 실행 진입점

보고서 기반 시뮬레이션 실행:
1. 설정 로드 (config.py)
2. Orchestrator 초기화
3. 실험 실행 (Run)
"""
import sys
import os

# 모듈 경로 추가 (필수)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.config import DEFAULT_CONFIG
from simulation.orchestrator import Orchestrator

def main():
    print("=== EthicaAI 시뮬레이션 시작 ===")
    
    # 설정 (기본값 사용)
    config = DEFAULT_CONFIG
    
    # 오케스트레이터 생성 및 실행
    orchestrator = Orchestrator(config)
    orchestrator.initialize()
    orchestrator.run()
    
    print("=== EthicaAI 시뮬레이션 종료 ===")

if __name__ == "__main__":
    main()
