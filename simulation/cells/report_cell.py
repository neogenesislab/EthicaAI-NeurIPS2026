"""
ReportCell — 실험 리포트 생성 및 저장 셀

역할:
    - EvalCell(통계)과 CausalCell(인과)의 분석 결과를 종합
    - JSON/CSV 파일로 결과 저장 (output_dir)
    - 콘솔에 요약 정보 출력
    - 논문용 그래프 생성을 위한 데이터 포맷팅
    
입력(Input):
    - metrics: Dict
    - hypothesis_test: Dict
    - causal_analysis: Dict
    - experiment_config: Dict

출력(Output):
    - report_path: str
"""
import os
import json
import datetime
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from simulation.cells.base_cell import BaseCell
from simulation.config import ExperimentConfig

class ReportCell(BaseCell):
    """리포트 생성 셀."""

    def __init__(self):
        super().__init__()
        self.output_dir = "simulation/outputs"

    @property
    def name(self) -> str:
        return "ReportCell"

    def initialize(self, config: Dict[str, Any]) -> None:
        super().initialize(config)
        self.output_dir = config.get("output_dir", "simulation/outputs")
        # output_dir이 절대 경로가 아닐 경우 프로젝트 루트 기준 처리 필요
        if not os.path.isabs(self.output_dir):
             # simulation/.. 로 이동하여 루트 기준 (상대 경로 문제 해결)
             # simulation/cells/report_cell.py -> simulation/cells -> simulation -> root
             current_file = os.path.abspath(__file__)
             simulation_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
             self.output_dir = os.path.join(simulation_dir, self.output_dir)
             
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """리포트 생성 및 저장."""
        metrics = input_data.get("metrics", {})
        hypothesis = input_data.get("hypothesis_test", {})
        causal = input_data.get("causal_analysis", {})
        config_summary = input_data.get("config_summary", {})
        trajectories = input_data.get("trajectories", [])
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 그래프 생성
        plot_paths = self._create_plots(trajectories, timestamp)
        
        report_data = {
            "timestamp": timestamp,
            "config": config_summary,
            "metrics": metrics,
            "hypothesis_test": hypothesis,
            "causal_analysis": causal,
            "plots": plot_paths
        }
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=4, ensure_ascii=False)
                
            self._print_console_summary(report_data)
            
            return {
                "report_path": filepath,
                "plot_paths": plot_paths,
                "status": "saved"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _create_plots(self, trajectories: List[Dict], timestamp: str) -> Dict[str, str]:
        """시뮬레이션 결과 그래프 생성."""
        if not trajectories:
            return {}
            
        plot_paths = {}
        df = pd.DataFrame(trajectories)
        
        # 데이터프레임 구조: [step, agent_id, obs, action, reward, ...]
        
        # 1. 누적 보상 그래프
        if "reward" in df.columns and "agent_id" in df.columns:
            plt.figure(figsize=(10, 6))
            
            # 피벗 테이블: index=step, columns=agent_id, values=reward
            # step별 보상 합계 (혹은 해당 스텝의 보상)
            pivot_df = df.pivot_table(index="step", columns="agent_id", values="reward", aggfunc="sum").fillna(0)
            cum_rewards = pivot_df.cumsum()
            
            for agent in cum_rewards.columns:
                plt.plot(cum_rewards.index, cum_rewards[agent], label=agent)
                    
            plt.title("Cumulative Rewards over Time")
            plt.xlabel("Step")
            plt.ylabel("Cumulative Reward")
            plt.legend()
            plt.grid(True)
            
            plot_filename = f"plot_rewards_{timestamp}.png"
            plot_path = os.path.join(self.output_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            plot_paths["cumulative_rewards"] = plot_path

        # 2. 행동 분포 (Actions) - 산점도
        if "action" in df.columns:
            plt.figure(figsize=(10, 6))
            
            agents = df["agent_id"].unique()
            for i, agent in enumerate(agents):
                agent_df = df[df["agent_id"] == agent]
                
                # 가독성을 위해 점을 찍되 약간의 노이즈(Jitter) 추가
                jitter = np.random.normal(0, 0.05, size=len(agent_df))
                plt.scatter(agent_df["step"], agent_df["action"] + jitter + (i * 0.1), label=agent, alpha=0.5, s=10)
                    
            plt.title("Agent Actions (0:C, 1:D, 2:C+P, 3:D+P)")
            plt.xlabel("Step")
            plt.ylabel("Action Type")
            plt.yticks([0, 1, 2, 3], ["Coop", "Defect", "Coop+Punish", "Defect+Punish"])
            plt.legend()
            plt.grid(True, axis='y')
            
            plot_filename = f"plot_actions_{timestamp}.png"
            plot_path = os.path.join(self.output_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            plot_paths["actions"] = plot_path
            
        return plot_paths

    def _print_console_summary(self, data: Dict[str, Any]):
        """콘솔 출력."""
        metrics = data.get("metrics", {})
        hypo = data.get("hypothesis_test", {})
        causal = data.get("causal_analysis", {})
        
        print("\n" + "="*40)
        print(f"   EthicaAI 실험 리포트   ")
        print("="*40)
        print(f"시간: {data['timestamp']}")
        print("-" * 20)
        print(f"[성과 지표]")
        print(f"  - Optimal 평균 보상: {metrics.get('avg_reward_optimal', 0):.2f}")
        print(f"  - Selfish 평균 보상: {metrics.get('avg_reward_selfish', 0):.2f}")
        print(f"  - 협력률: {metrics.get('cooperation_rate', 0)*100:.1f}%")
        print("-" * 20)
        print(f"[가설 검증 (Optimal > Selfish)]")
        print(f"  - p-value: {hypo.get('p_value', 1.0):.4f}")
        print(f"  - 유의함: {'예' if hypo.get('significant') else '아니오'}")
        print("-" * 20)
        print(f"[인과 분석 (헌신 -> 보상)]")
        print(f"  - ATE (인과 효과): {causal.get('ate_estimate', 0):.4f}")
        print(f"  - 상관계수: {causal.get('correlation', 0):.4f}")
        print("="*40 + "\n")
