"""
ResearchDirector: EthicaAI 자율 연구소의 최상위 의사결정 에이전트.

역할:
- 연구 의제(Research Agenda) 관리
- 성공 시 후속 연구 자동 생성
- 실패 시 대안 연구로 전환
- LLM을 활용한 새 연구 질문 자동 생성
- v2.0: 트리 탐색 모드 지원 (AgenticTreeSearch 연동)
"""

import os
import json
from datetime import datetime

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class ResearchDirector:
    """
    5번째 에이전트: 연구 의제를 관리하는 최상위 의사결정자.
    "성공하면 더 깊이, 실패하면 대안을 — 끝없이 연구하는 AI 연구소"
    """

    def __init__(
        self,
        agenda_path="experiments/evolution/research_agenda.json",
        history_path="experiments/evolution/history.json",
        tree_search=None,
    ):
        self.agenda_path = agenda_path
        self.history_path = history_path
        self.tree_search = tree_search  # v2.0: AgenticTreeSearch 인스턴스
        self.agenda = self._load_agenda()

        # LLM 초기화 (Theorist와 동일한 방식)
        self.model = None
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            ".env",
        )
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("GEMINI_API_KEY="):
                        os.environ["GEMINI_API_KEY"] = line.strip().split("=", 1)[1]
                        break

        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and genai:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")

    # ─────────────────────────────────────────────
    # 핵심 메서드
    # ─────────────────────────────────────────────

    def get_active_question(self):
        """현재 활성화된 연구 과제를 반환. 없으면 None."""
        questions = self.agenda.get("questions", {})

        # 1. 이미 active인 과제가 있으면 그것을 반환
        for q in questions.values():
            if q["status"] == "active":
                return q

        # 2. 없으면, queued 중 의존성 충족 + 최고 우선순위를 활성화
        candidates = []
        for q in questions.values():
            if q["status"] != "queued":
                continue
            # 의존성 검사
            dep = q.get("depends_on")
            if dep:
                dep_q = questions.get(dep)
                if not dep_q or dep_q["status"] != "completed":
                    continue  # 의존 과제 미완료 → 건너뜀
            candidates.append(q)

        if not candidates:
            return None

        # 우선순위 기준 정렬 (낮은 숫자 = 높은 우선순위)
        candidates.sort(key=lambda x: x.get("priority", 99))
        chosen = candidates[0]
        chosen["status"] = "active"
        self._save_agenda()
        print(f"📋 [Director] 새 연구 과제 활성화: {chosen['id']} — {chosen['question']}")
        return chosen

    def on_generation_complete(self, question_id, results, config):
        """
        한 세대 완료 후 호출.
        반환값:
        - "continue": 같은 과제 계속
        - "success": 목표 달성
        - "pivot": 실패 횟수 초과, 대안으로 전환
        """
        question = self.agenda["questions"].get(question_id)
        if not question:
            return "continue"

        # 세대 카운트 증가
        question["generation_count"] = question.get("generation_count", 0) + 1
        self.agenda["total_generations_run"] = self.agenda.get("total_generations_run", 0) + 1

        # 최고 결과 갱신
        coop = results.get("Prosocial", {}).get("cooperation_rate", 0.0)
        current_best = question.get("best_result") or 0.0
        if coop > current_best:
            question["best_result"] = coop
            question["best_config"] = config

        # 성공 판정
        criteria = question.get("success_criteria", {})
        target = criteria.get("target", 0.5)
        condition = criteria.get("condition", ">")

        is_success = False
        if condition == ">":
            is_success = coop > target
        elif condition == ">=":
            is_success = coop >= target

        if is_success:
            self._save_agenda()
            return "success"

        # 실패: 최대 세대 초과 확인
        max_gen = question.get("constraints", {}).get("max_generations", 50)
        if question["generation_count"] >= max_gen:
            self._save_agenda()
            return "pivot"

        self._save_agenda()
        return "continue"

    def on_success(self, question_id, best_config):
        """
        연구 과제 성공 처리:
        1. 현재 과제를 'completed'로 마킹
        2. 정적 후속 과제 활성화
        3. LLM으로 동적 후속 과제 생성
        """
        question = self.agenda["questions"][question_id]
        question["status"] = "completed"
        question["outcome"] = "success"
        question["completed_at"] = datetime.now().isoformat()
        self.agenda["total_questions_completed"] = (
            self.agenda.get("total_questions_completed", 0) + 1
        )

        print(f"\n🎉 [Director] 연구 성공! {question_id}: {question['question']}")
        print(f"   최고 협력률: {question.get('best_result', 0):.4f}")
        print(f"   소요 세대: {question.get('generation_count', 0)}")

        # 이벤트 기록
        self._log_event("question_completed", question_id, "success", best_config)

        # 정적 후속 과제 활성화
        on_success = question.get("on_success", {})
        if on_success.get("action") == "spawn":
            for next_id in on_success.get("questions", []):
                if next_id in self.agenda["questions"]:
                    next_q = self.agenda["questions"][next_id]
                    if next_q["status"] == "queued":
                        print(f"   → 후속 과제 대기: {next_id}")

        # LLM으로 동적 후속 과제 생성
        if self.model:
            try:
                new_questions = self._generate_followup(question)
                for nq in new_questions:
                    self.agenda["questions"][nq["id"]] = nq
                    print(f"   → 🤖 AI 생성 후속 과제: {nq['id']} — {nq['question']}")
            except Exception as e:
                print(f"   ⚠️ 후속 과제 자동 생성 실패: {e}")

        # v2.0: 트리 탐색 모드 — 성공한 설정을 루트로 깊이 탐색
        if self.tree_search:
            try:
                root = self.tree_search.create_root(
                    best_config, hypothesis=f"Success on {question_id}"
                )
                children = self.tree_search.expand(root, num_children=2)
                for child in children:
                    qid = self._next_question_id()
                    nq_data = {
                        "question": child.hypothesis,
                        "question_kr": child.hypothesis_kr,
                        "type": "tree_exploration",
                    }
                    full_q = self._make_question(qid, nq_data, parent=question_id)
                    full_q["tree_node_id"] = child.node_id
                    self.agenda["questions"][qid] = full_q
                    print(f"   → 🌳 트리 탐색 과제: {qid} — {child.hypothesis_kr}")
                self.tree_search.save_tree()
            except Exception as e:
                print(f"   ⚠️ 트리 탐색 확장 실패: {e}")

        self._save_agenda()

    def on_failure(self, question_id):
        """
        연구 과제 실패 처리:
        1. retry_count 증가
        2. max_retries 초과 시 fallback으로 전환
        3. fallback 없으면 LLM에게 대안 생성 요청
        """
        question = self.agenda["questions"][question_id]
        question["retry_count"] = question.get("retry_count", 0) + 1

        on_failure = question.get("on_failure", {})
        max_retries = on_failure.get("max_retries", 2)

        print(f"\n❌ [Director] 연구 실패: {question_id} (시도 {question['retry_count']}/{max_retries})")
        print(f"   최고 결과: {question.get('best_result', 0):.4f}")

        if question["retry_count"] < max_retries:
            # 재시도: 히스토리 리셋 후 같은 과제 계속
            question["generation_count"] = 0
            print(f"   → 재시도합니다 (히스토리 리셋).")
            if os.path.exists(self.history_path):
                os.remove(self.history_path)
            self._save_agenda()
            return

        # 최대 재시도 초과 → 과제 종료
        question["status"] = "failed"
        question["outcome"] = "failure"
        question["completed_at"] = datetime.now().isoformat()
        self._log_event("question_failed", question_id, "failure", None)

        # Fallback 과제로 전환
        action = on_failure.get("action", "archive")
        fallback_id = on_failure.get("fallback_to")

        if action == "fallback" and fallback_id:
            if fallback_id in self.agenda["questions"]:
                fb = self.agenda["questions"][fallback_id]
                if fb["status"] == "queued":
                    fb["status"] = "active"
                    print(f"   → 대안 과제로 전환: {fallback_id} — {fb['question']}")
        elif action == "generate_new" and self.model:
            # LLM에게 대안 생성 요청
            try:
                alt = self._generate_alternative(question)
                self.agenda["questions"][alt["id"]] = alt
                alt["status"] = "active"
                print(f"   → 🤖 AI 생성 대안 과제: {alt['id']} — {alt['question']}")
            except Exception as e:
                print(f"   ⚠️ 대안 자동 생성 실패: {e}")
        else:
            print(f"   → 과제 종료 (아카이브됨).")

        self._save_agenda()

    def generate_new_questions(self):
        """모든 과제가 소진되었을 때, LLM에게 새 연구 질문 생성을 요청."""
        if not self.model:
            print("⚠️ [Director] LLM 없이는 새 질문을 생성할 수 없습니다.")
            return

        # 지금까지의 연구 성과 요약
        completed = [
            q for q in self.agenda["questions"].values() if q["status"] == "completed"
        ]
        failed = [
            q for q in self.agenda["questions"].values() if q["status"] == "failed"
        ]

        summary = {
            "completed": [
                {"question": q["question"], "result": q.get("best_result")} for q in completed
            ],
            "failed": [
                {"question": q["question"], "result": q.get("best_result")} for q in failed
            ],
        }

        prompt = f"""
You are the Research Director of EthicaAI Genesis Lab.

ALL research questions have been exhausted. Here is the research history:
{json.dumps(summary, indent=2)}

Based on what we've learned, propose 2 entirely NEW research directions.
Consider:
1. What patterns emerged from successes and failures?
2. What fundamental assumptions haven't been tested?
3. What would be a breakthrough discovery?

Output JSON array:
[
  {{
    "question": "A concise research question in English",
    "question_kr": "한국어 버전",
    "type": "exploration",
    "success_criteria": {{"metric": "cooperation_rate", "condition": ">", "target": 0.5}},
    "rationale": "Why this question matters",
    "rationale_kr": "왜 이 질문이 중요한지 한국어 설명"
  }}
]
"""
        try:
            response = self.model.generate_content(prompt)
            text = response.text.replace("```json", "").replace("```", "").strip()
            new_qs = json.loads(text)

            for i, nq in enumerate(new_qs):
                qid = self._next_question_id()
                full_q = self._make_question(qid, nq)
                self.agenda["questions"][qid] = full_q
                print(f"🤖 [Director] 새 연구 방향 생성: {qid} — {full_q['question']}")

            self._save_agenda()
        except Exception as e:
            print(f"⚠️ [Director] 새 질문 생성 실패: {e}")

    def get_progress_summary(self):
        """대시보드용 연구 진행 상황 요약."""
        questions = self.agenda.get("questions", {})
        total = len(questions)
        completed = sum(1 for q in questions.values() if q["status"] == "completed")
        failed = sum(1 for q in questions.values() if q["status"] == "failed")
        active = [q for q in questions.values() if q["status"] == "active"]

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "active": active[0] if active else None,
            "total_generations": self.agenda.get("total_generations_run", 0),
            "progress_pct": (completed / total * 100) if total > 0 else 0,
        }

    # ─────────────────────────────────────────────
    # LLM 기반 과제 자동 생성
    # ─────────────────────────────────────────────

    def _generate_followup(self, completed_question):
        """성공한 과제를 기반으로 후속 연구 과제 생성."""
        prompt = f"""
You are the Research Director of EthicaAI Genesis Lab.

A research question has been SUCCESSFULLY answered:
- Question: {completed_question['question']}
- Best Cooperation Rate: {completed_question.get('best_result', 0):.4f}
- Generations Used: {completed_question.get('generation_count', 0)}

Based on this success, propose 2 follow-up research questions.
Consider:
1. Can we push the result higher? (Goal escalation)
2. Does this generalize? (Robustness test)
3. WHY did it work? (Mechanistic understanding)

Output JSON array:
[
  {{
    "question": "A concise research question in English",
    "question_kr": "한국어 버전",
    "type": "escalation|generalization|analysis",
    "success_criteria": {{"metric": "cooperation_rate", "condition": ">", "target": float}},
    "rationale": "Why this question matters",
    "rationale_kr": "왜 이 질문이 중요한지 한국어 설명"
  }}
]
"""
        response = self.model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        raw_questions = json.loads(text)

        result = []
        for nq in raw_questions:
            qid = self._next_question_id()
            full_q = self._make_question(qid, nq, parent=completed_question["id"])
            result.append(full_q)
        return result

    def _generate_alternative(self, failed_question):
        """실패한 과제에 대한 대안 연구 과제 생성."""
        prompt = f"""
You are the Research Director of EthicaAI Genesis Lab.

A research question has FAILED after {failed_question.get('generation_count', 0)} generations:
- Question: {failed_question['question']}
- Best result: {failed_question.get('best_result', 0):.4f}
- Target was: {failed_question.get('success_criteria', {}).get('target', 0.5)}

The current approach isn't working. Propose 1 ALTERNATIVE approach.
Consider:
1. Changing the environment or reward structure
2. A completely different optimization strategy
3. Relaxing constraints or reframing the problem

Output JSON:
{{
  "question": "A concise alternative research question",
  "question_kr": "한국어 버전",
  "type": "pivot",
  "success_criteria": {{"metric": "cooperation_rate", "condition": ">", "target": float}},
  "rationale": "Why this alternative might work",
  "rationale_kr": "왜 이 대안이 효과적일 수 있는지 한국어 설명"
}}
"""
        response = self.model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        raw = json.loads(text)

        qid = self._next_question_id()
        return self._make_question(qid, raw, parent=failed_question["id"])

    # ─────────────────────────────────────────────
    # 보조 메서드
    # ─────────────────────────────────────────────

    def _load_agenda(self):
        """연구 의제를 JSON에서 로드."""
        if os.path.exists(self.agenda_path):
            with open(self.agenda_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # 파일 없으면 빈 의제 생성
        return {
            "lab_name": "EthicaAI Genesis Lab",
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "total_generations_run": 0,
            "total_questions_completed": 0,
            "questions": {},
            "history": [],
        }

    def _save_agenda(self):
        """연구 의제를 JSON으로 저장."""
        os.makedirs(os.path.dirname(self.agenda_path), exist_ok=True)
        with open(self.agenda_path, "w", encoding="utf-8") as f:
            json.dump(self.agenda, f, indent=2, ensure_ascii=False)

    def _log_event(self, event_type, question_id, outcome, config):
        """연구 이벤트를 히스토리에 기록."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "question_id": question_id,
            "outcome": outcome,
        }
        if config:
            event["best_config"] = {
                k: v for k, v in config.items() if k.startswith("GENESIS_")
            }
        self.agenda.setdefault("history", []).append(event)

    def _next_question_id(self):
        """다음 연구 과제 ID 생성 (RQ-XXX)."""
        existing_ids = list(self.agenda.get("questions", {}).keys())
        if not existing_ids:
            return "RQ-001"
        max_num = max(int(qid.split("-")[1]) for qid in existing_ids if qid.startswith("RQ-"))
        return f"RQ-{max_num + 1:03d}"

    def _make_question(self, qid, raw, parent=None):
        """LLM 출력을 정규 연구 과제 구조로 변환."""
        criteria = raw.get("success_criteria", {"metric": "cooperation_rate", "condition": ">", "target": 0.5})

        return {
            "id": qid,
            "question": raw.get("question_kr", raw.get("question", "자동 생성된 연구 과제")),
            "question_en": raw.get("question", "Auto-generated research question"),
            "type": raw.get("type", "exploration"),
            "status": "queued",
            "priority": 2,
            "created_at": datetime.now().isoformat(),
            "success_criteria": criteria,
            "constraints": {
                "max_generations": 40,
                "parameter_space": {
                    "GENESIS_BETA": [0.01, 100.0],
                    "GENESIS_ALPHA": [0.01, 5.0],
                    "GENESIS_LOGIC_MODE": ["adaptive_beta", "inverse_beta", "institutional"],
                    "IA_ALPHA": [0.1, 10.0],
                    "IA_BETA": [0.01, 1.0],
                    "USE_INEQUITY_AVERSION": [True, False],
                },
            },
            "on_success": {"action": "spawn", "questions": []},
            "on_failure": {
                "action": "generate_new",
                "fallback_to": None,
                "max_retries": 1,
            },
            "depends_on": None,
            "parent": parent,
            "generation_count": 0,
            "retry_count": 0,
            "best_result": None,
            "completed_at": None,
            "outcome": None,
        }


if __name__ == "__main__":
    director = ResearchDirector()
    summary = director.get_progress_summary()
    print(f"📊 연구소 현황: {json.dumps(summary, indent=2, ensure_ascii=False, default=str)}")

    q = director.get_active_question()
    if q:
        print(f"📋 현재 활성 과제: {q['id']} — {q['question']}")
    else:
        print("📋 활성 과제 없음. 새 질문 생성 중...")
        director.generate_new_questions()
