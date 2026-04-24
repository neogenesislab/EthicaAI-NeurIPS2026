"""
EthicaAI 셀 기반 에이전트 유기체 아키텍처 — 셀 패키지 초기화

각 셀은 독립 모듈이면서 Orchestrator가 조율하여
하나의 유기체처럼 동작한다.

실행 순서: EnvCell → AgentCell → TrainCell → EvalCell → CausalCell → ReportCell
"""
from simulation.cells.base_cell import BaseCell

__all__ = ["BaseCell"]
