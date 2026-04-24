"""
Orchestrator — 시뮬레이션 파이프라인 조율자

역할:
    - 모든 셀(EnvCell, AgentCell, TrainCell, EvalCell, CausalCell, ReportCell)의 생명주기 관리
    - 셀 간 데이터 흐름 제어 (Env -> Agent -> Train -> ...)
    - 실험 루프 실행 (Episode/Step Loop)

설계 원칙:
    - 각 셀은 서로 직접 통신하지 않고 Orchestrator를 통해서만 데이터를 주고받음.
    - 단계별 실행 로그 기록.
"""
import time
from typing import Any, Dict, List

from simulation.config import ExperimentConfig
from simulation.cells.env_cell import EnvCell
from simulation.cells.agent_cell import AgentCell
from simulation.cells.train_cell import TrainCell
from simulation.cells.eval_cell import EvalCell
from simulation.cells.causal_cell import CausalCell
from simulation.cells.report_cell import ReportCell


class Orchestrator:
    """실험 조율자."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # 셀 인스턴스화
        self.env_cell = EnvCell()
        self.agent_cell = AgentCell()
        self.train_cell = TrainCell()
        self.eval_cell = EvalCell()
        self.causal_cell = CausalCell()
        self.report_cell = ReportCell()
        
        self.cells = [
            self.env_cell,
            self.agent_cell,
            self.train_cell,
            self.eval_cell,
            self.causal_cell,
            self.report_cell,
        ]

    def initialize(self):
        """모든 셀 초기화."""
        print("[Orchestrator] 초기화 시작...")
        
        env_config = {
            "env_config": self.config.prisoners_dilemma,
            "seed": self.config.seed
        }
        self.env_cell.initialize(env_config)
        
        agent_config = {
            "agent_config": self.config.agent,
            "env_agents": self.env_cell.env.possible_agents
        }
        self.agent_cell.initialize(agent_config)
        
        train_config = {
            "train_config": self.config.train
        }
        self.train_cell.initialize(train_config)
        
        eval_config = {"stats_config": self.config.statistics}
        self.eval_cell.initialize(eval_config)
        
        causal_config = {"causal_config": self.config.causal}
        self.causal_cell.initialize(causal_config)
        
        report_config = {"output_dir": self.config.output_dir}
        self.report_cell.initialize(report_config)
        
        print("[Orchestrator] 초기화 완료.")

    def run(self):
        """메인 실험 루프 실행."""
        print(f"[Orchestrator] 실험 '{self.config.experiment_name}' 시작 (라운드: {self.config.prisoners_dilemma.num_rounds})")
        
        # 1. 환경 리셋
        env_output = self.env_cell.execute({"reset": True})
        current_obs = env_output["observations"]
        next_agent = env_output.get("next_agent") # AEC 환경
        
        trajectories = [] # 학습용 데이터 수집
        
        # 2. 메인 루프 (Step 단위)
        # 최대 스텝 수 제한 (무한 루프 방지)
        max_steps = self.config.prisoners_dilemma.num_rounds * len(self.env_cell.env.possible_agents)
        
        start_time = time.time()
        
        for step in range(max_steps):
            # 2-1. 에이전트 행동 결정
            agent_input = {
                "observations": current_obs,
                "next_agent": next_agent
            }
            agent_output = self.agent_cell.execute(agent_input)
            actions = agent_output["actions"]
            
            if not actions:
                print(f"[Orchestrator] 행동 선택 불가 (Step {step}). 종료.")
                break
                
            # 2-2. 환경 스텝 실행
            env_input = {
                "actions": actions
            }
            env_output = self.env_cell.execute(env_input)
            
            next_obs = env_output["observations"]
            rewards = env_output["rewards"]
            terminations = env_output["terminations"]
            truncations = env_output["truncations"]
            next_agent = env_output.get("next_agent") # 다음 턴 에이전트
            
            # 2-3. 데이터 수집 (Trajectory)
            # 여기서는 AEC 구조라 복잡함. (행동한 에이전트만 보상 받음)
            # 간단히 행동한 에이전트에 대해 기록
            for agent_id, action in actions.items():
                reward = rewards.get(agent_id, 0.0)
                # 에이전트 내부 상태 업데이트 (학습용 아님, 단순 누적)
                if agent_id in self.agent_cell.agents:
                     self.agent_cell.agents[agent_id].update(reward)
                
                traj = {
                    "step": step,
                    "agent_id": agent_id,
                    "obs": current_obs.get(agent_id),
                    "action": action,
                    "reward": reward,
                    "next_obs": next_obs.get(agent_id),
                    "done": terminations.get(agent_id, False) or truncations.get(agent_id, False)
                }
                trajectories.append(traj)

            # 상태 업데이트
            current_obs = next_obs
            
            # 2-4. 주기적 학습 (TrainCell)
            if step > 0 and step % self.config.train.log_interval == 0:
                print(f"[Orchestrator] Step {step}/{max_steps} 진행 중...")
                train_input = {"trajectories": trajectories}
                train_output = self.train_cell.execute(train_input)
                # trajectories = [] # 버퍼 비우기 (On-policy)
                
            # 2-5. 종료 조건 확인
            if all(terminations.values()) or all(truncations.values()):
                print(f"[Orchestrator] 모든 에이전트 종료 (Step {step})")
                break
                
        end_time = time.time()
        print(f"[Orchestrator] 실험 종료. 소요 시간: {end_time - start_time:.2f}초")
        
        # 3. 평가 및 리포트 (Eval -> Causal -> Report)
        
        # 에이전트별 총 보상 데이터 수집
        agents_data = {
            aid: agent.total_reward 
            for aid, agent in self.agent_cell.agents.items()
        }
        
        # 3-1. 평가 (EvalCell)
        eval_input = {
            "trajectories": trajectories,
            "agents_data": agents_data
        }
        eval_output = self.eval_cell.execute(eval_input)
        
        # 3-2. 인과 분석 (CausalCell)
        causal_input = {
            "agents_data": agents_data,
            # agent_config 정보가 필요하다면 여기서 전달
        }
        causal_output = self.causal_cell.execute(causal_input)
        
        # 3-3. 리포트 생성 (ReportCell)
        report_input = {
            "metrics": eval_output.get("metrics"),
            "hypothesis_test": eval_output.get("hypothesis_test"),
            "causal_analysis": causal_output,
            "config_summary": {
                "name": self.config.experiment_name,
                "rounds": self.config.prisoners_dilemma.num_rounds,
                "agents": len(self.agent_cell.agents)
            },
            "trajectories": trajectories
        }
        self.report_cell.execute(report_input)
        
        # 최종 결과 요약 (간단 버전)
        self._print_summary()

    def _print_summary(self):
        """결과 출력."""
        print("\n=== 실험 결과 요약 ===")
        for agent_id, agent in self.agent_cell.agents.items():
            print(f"- {agent_id} ({agent.__class__.__name__}): 총 보상 = {agent.total_reward:.2f}")
            if hasattr(agent, "current_mode"):
                print(f"  - 마지막 모드: {agent.current_mode}")
        print("======================")
