"""
EthicaAI Genesis Lab v2.0 — 자율 연구소 메인 엔트리포인트.

이중 루프 구조:
- 외부 루프: Research Director가 연구 과제를 관리
- 내부 루프: Theorist → Engineer → Critic 사이클
- v2.0: IA 보상 변환 + 3단계 Coordinator 개입 + 트리 탐색

"성공하면 더 깊이, 실패하면 대안을 — 끝없이 연구하는 AI 연구소"
"""

import os
import sys
import json
import time

# 루트 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulation.genesis.theorist import Theorist
from simulation.genesis.engineer import Engineer
from simulation.genesis.critic import Critic
from simulation.genesis.coordinator import Coordinator
from simulation.genesis.research_director import ResearchDirector


def run_research_lab():
    """EthicaAI 자율 연구소 메인 루프."""

    print("=" * 60)
    print("🏛️  EthicaAI Genesis Lab — Autonomous R&D System")
    print("   '성공하면 더 깊이, 실패하면 대안을'")
    print("=" * 60)

    # 에이전트 초기화
    director = ResearchDirector()
    theorist = Theorist()
    engineer = Engineer()
    critic = Critic()
    coordinator = Coordinator()

    # CSV 로그 초기화
    csv_path = "experiments/evolution/evolution_progress.csv"
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w") as f:
            f.write("Generation,Beta,Alpha,Mode,Coop_Prosocial,Coop_Individualist,Success,QuestionID\n")

    lab_start_time = time.time()

    # ═══════════════════════════════════════════
    # 외부 루프: 연구 과제 단위
    # ═══════════════════════════════════════════
    while True:
        question = director.get_active_question()

        if question is None:
            # 모든 정적 과제 소진 → LLM에게 새 질문 요청
            print("\n🔍 [Lab] 모든 연구 과제 소진. AI에게 새 방향을 요청합니다...")
            director.generate_new_questions()
            question = director.get_active_question()

            if question is None:
                print("\n🏁 [Lab] 더 이상 연구할 과제가 없습니다. 연구소 종료.")
                break

        qid = question["id"]
        criteria = question.get("success_criteria", {})

        print("\n" + "─" * 60)
        print(f"📋 연구 과제: {qid}")
        print(f"   질문: {question['question']}")
        print(f"   목표: {criteria.get('metric', 'cooperation_rate')} {criteria.get('condition', '>')} {criteria.get('target', 0.5)}")
        print(f"   유형: {question.get('type', 'unknown')}")
        print("─" * 60)

        generation = 0

        # ═══════════════════════════════════════
        # 내부 루프: Theorist → Engineer → Critic
        # ═══════════════════════════════════════
        while True:
            generation += 1
            print(f"\n🔄 --- [{qid}] Generation {generation} ---")

            # 0. Coordinator: v2.0 3단계 개입 시스템
            is_stagnant, msg = coordinator.check_stagnation()
            if is_stagnant:
                print(f"🚨 {msg}")
                # 정체 횟수에 따라 개입 수준 결정
                stagnation_count = getattr(coordinator, '_stagnation_count', 0) + 1
                coordinator._stagnation_count = stagnation_count
                
                if stagnation_count <= 2:
                    coordinator.intervene(method="poke")
                    print("  > Level 1 (Poke): IA 파라미터 조정")
                elif stagnation_count <= 4:
                    coordinator.intervene(method="shock")
                    print("  > Level 2 (Shock): 로직 모드 전환")
                else:
                    coordinator.intervene(method="reset")
                    coordinator._stagnation_count = 0
                    print("  > Level 3 (Reset): 히스토리 초기화")

            # 1. Theorist: 다음 설정 제안
            print("🧠 Theorist is thinking...")
            try:
                next_config, proposal = theorist.propose_next_config()
            except Exception as e:
                print(f"⚠️ Theorist Error: {e}. Using fallback.")
                next_config = {
                    "GENESIS_BETA": 0.1,
                    "GENESIS_ALPHA": 0.1,
                    "GENESIS_LOGIC_MODE": "adaptive_beta",
                }
                proposal = {}

            # Config 저장 + v2.0 IA 파라미터 강제 활성화
            next_config["rationale"] = proposal.get("rationale", "")
            next_config["rationale_kr"] = proposal.get("rationale_kr", "")
            
            # v2.0: IA 보상 변환 강제 활성화 (기본값이 False이므로)
            next_config.setdefault("USE_INEQUITY_AVERSION", True)
            next_config.setdefault("IA_ALPHA", 5.0)
            next_config.setdefault("IA_BETA", 0.05)
            next_config.setdefault("IA_EMA_LAMBDA", 0.95)
            
            os.makedirs("experiments/evolution/current", exist_ok=True)
            with open("experiments/evolution/current/config.json", "w") as f:
                json.dump(next_config, f, indent=4)
            print(f"  > Proposed: Beta={next_config.get('GENESIS_BETA')}, "
                  f"Alpha={next_config.get('GENESIS_ALPHA')}, "
                  f"Mode={next_config.get('GENESIS_LOGIC_MODE')}, "
                  f"IA={'ON' if next_config.get('USE_INEQUITY_AVERSION') else 'OFF'}")

            # 2. Engineer: 시뮬레이션 실행
            print("🛠️ Engineer is running simulation...")
            start_time = time.time()
            try:
                results = engineer.run_simulation()
            except Exception as e:
                print(f"⚠️ Engineer Error: {e}. Skipping this generation.")
                results = {
                    "Prosocial": {"cooperation_rate": 0.0, "reward_mean": 0.0, "gini": 0.0},
                    "Individualist": {"cooperation_rate": 0.0, "reward_mean": 0.0, "gini": 0.0},
                }
            elapsed = time.time() - start_time
            print(f"  > Simulation finished in {elapsed:.1f}s")

            # 3. Critic: 결과 평가 (동적 성공 기준)
            print("🧐 Critic is analyzing...")
            success = critic.evaluate(success_criteria=criteria)

            # CSV 로그 기록
            prosocial_coop = results.get("Prosocial", {}).get("cooperation_rate", 0.0)
            individualist_coop = results.get("Individualist", {}).get("cooperation_rate", 0.0)

            with open(csv_path, "a") as f:
                f.write(
                    f"{generation},"
                    f"{next_config.get('GENESIS_BETA')},"
                    f"{next_config.get('GENESIS_ALPHA')},"
                    f"{next_config.get('GENESIS_LOGIC_MODE')},"
                    f"{prosocial_coop:.4f},"
                    f"{individualist_coop:.4f},"
                    f"{success},"
                    f"{qid}\n"
                )

            # 4. Research Director: 세대별 의사결정
            decision = director.on_generation_complete(qid, results, next_config)

            if decision == "success":
                print(f"\n🎉 [{qid}] 목표 달성!")
                director.on_success(qid, next_config)
                break

            elif decision == "pivot":
                print(f"\n🔀 [{qid}] 최대 세대 도달. 전환합니다.")
                director.on_failure(qid)
                break

            else:
                # "continue" → 다음 세대
                print(f"❌ [{qid}] Coop={prosocial_coop:.4f} (목표: {criteria.get('target', 0.5)}). 계속...")

    # 연구소 종료 보고
    total_time = time.time() - lab_start_time
    summary = director.get_progress_summary()

    print("\n" + "=" * 60)
    print("🏛️  EthicaAI Genesis Lab — 연구 종료 보고서")
    print("=" * 60)
    print(f"  총 소요 시간: {total_time / 60:.1f}분")
    print(f"  총 세대 실행: {summary['total_generations']}")
    print(f"  연구 과제: {summary['completed']} 완료 / {summary['failed']} 실패 / {summary['total']} 전체")
    print(f"  진행률: {summary['progress_pct']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_research_lab()
