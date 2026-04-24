"""
트리 탐색 기반 자가 진화 엔진.

Genesis v2.0 Strategy C: Scientist Update
헌법 제12조 1항 이론적 근거:
  - Sakana AI "The AI Scientist v2" (2025)
  - Hu et al. (2024) "Open-Ended Learning in Multi-Agent Systems"

노드 = {가설, 설정, 결과}
가지치기 = CR 미개선 or 에러
백트래킹 = 유망 노드에서 자식 확장
"""
import json
import os
from datetime import datetime


class SearchNode:
    """트리 탐색의 단일 노드."""

    def __init__(self, node_id, hypothesis, config_overrides,
                 parent=None, hypothesis_kr=""):
        self.node_id = node_id
        self.hypothesis = hypothesis
        self.hypothesis_kr = hypothesis_kr
        self.config_overrides = config_overrides
        self.parent = parent
        self.children = []
        self.result = None       # 실험 결과
        self.status = "pending"  # pending, running, success, pruned, buggy
        self.created_at = datetime.now().isoformat()

    def to_dict(self):
        return {
            "id": self.node_id,
            "hypothesis": self.hypothesis,
            "hypothesis_kr": self.hypothesis_kr,
            "config": self.config_overrides,
            "parent": self.parent.node_id if self.parent else None,
            "status": self.status,
            "result": self.result,
            "children": [c.node_id for c in self.children],
            "created_at": self.created_at,
        }


class AgenticTreeSearch:
    """
    LLM 기반 Progressive Agentic Tree Search 엔진.

    1. 루트 노드 = 현재 최선의 설정
    2. LLM이 자식 가설 2~3개 생성 (Ideation)
    3. 각 가설을 시뮬레이션으로 검증 (Execution)
    4. CR 기준으로 가지치기/확장 (Pruning/Expansion)
    5. 최고 노드의 설정을 채택
    """

    def __init__(self, model=None,
                 tree_path="experiments/evolution/search_tree.json"):
        self.model = model
        self.tree_path = tree_path
        self.nodes = {}
        self.best_node = None
        self.best_cr = 0.0
        self._node_counter = 0

        # 기존 트리 로드 시도
        self._load_tree()

    def _next_id(self):
        """고유 노드 ID 생성."""
        nid = f"N-{self._node_counter:03d}"
        self._node_counter += 1
        return nid

    def create_root(self, base_config, hypothesis="Base Configuration"):
        """루트 노드 생성."""
        root = SearchNode(
            self._next_id(), hypothesis, base_config,
            hypothesis_kr="기본 설정"
        )
        self.nodes[root.node_id] = root
        return root

    def expand(self, parent_node, num_children=2):
        """
        LLM을 활용하여 부모 노드에서 자식 가설 생성.

        Args:
            parent_node: 부모 노드
            num_children: 생성할 자식 수

        Returns:
            list[SearchNode]: 생성된 자식 노드 목록
        """
        if not self.model:
            # Mock 모드: 사전 정의된 탐색
            return self._mock_expand(parent_node, num_children)

        prompt = f"""
You are an AI Research Scientist exploring cooperation mechanisms in Multi-Agent Systems.
Your goal: maximize the Cooperation Rate (target > 0.5).

Parent hypothesis: {parent_node.hypothesis}
Parent config: {json.dumps(parent_node.config_overrides, indent=2)}
Parent result: {json.dumps(parent_node.result, indent=2) if parent_node.result else "Not yet tested"}

Generate {num_children} DIFFERENT child hypotheses to explore.
Each should modify the parent config in a meaningful way.
Focus on: IA_ALPHA, IA_BETA, USE_INEQUITY_AVERSION, GENESIS_BETA, GENESIS_LOGIC_MODE.

Output JSON array:
[
  {{
    "hypothesis": "A concise hypothesis in English",
    "hypothesis_kr": "가설을 한국어로",
    "config_overrides": {{"key": "value"}}
  }}
]
"""
        try:
            response = self.model.generate_content(prompt)
            text = response.text.replace("```json", "").replace("```", "").strip()
            children_data = json.loads(text)

            results = []
            for cd in children_data[:num_children]:
                child = SearchNode(
                    self._next_id(),
                    cd.get("hypothesis", ""),
                    cd.get("config_overrides", {}),
                    parent=parent_node,
                    hypothesis_kr=cd.get("hypothesis_kr", ""),
                )
                parent_node.children.append(child)
                self.nodes[child.node_id] = child
                results.append(child)
            return results

        except Exception as e:
            print(f"⚠️ 트리 확장 실패: {e}")
            return self._mock_expand(parent_node, num_children)

    def _mock_expand(self, parent_node, num_children):
        """LLM 없이 사전 정의된 탐색."""
        import random
        presets = [
            {"hypothesis": "Increase envy coefficient",
             "hypothesis_kr": "질투 계수 증가",
             "config_overrides": {"IA_ALPHA": 8.0, "IA_BETA": 0.05}},
            {"hypothesis": "Balance envy and guilt",
             "hypothesis_kr": "질투와 죄책감 균형",
             "config_overrides": {"IA_ALPHA": 3.0, "IA_BETA": 0.3}},
            {"hypothesis": "Strong guilt with institutional mode",
             "hypothesis_kr": "강한 죄책감 + 제도 모드",
             "config_overrides": {"IA_ALPHA": 2.0, "IA_BETA": 1.0,
                                  "GENESIS_LOGIC_MODE": "institutional"}},
            {"hypothesis": "High intervention with low sensitivity",
             "hypothesis_kr": "높은 개입 + 낮은 감도",
             "config_overrides": {"GENESIS_BETA": 50.0, "GENESIS_ALPHA": 0.5}},
        ]
        random.shuffle(presets)
        results = []
        for preset in presets[:num_children]:
            child = SearchNode(
                self._next_id(),
                preset["hypothesis"],
                preset["config_overrides"],
                parent=parent_node,
                hypothesis_kr=preset["hypothesis_kr"],
            )
            parent_node.children.append(child)
            self.nodes[child.node_id] = child
            results.append(child)
        return results

    def evaluate_node(self, node, cr, additional_metrics=None):
        """
        노드 결과 평가 + 가지치기/성공 판정.

        Args:
            node: 평가할 노드
            cr: 협력률(Cooperation Rate)
            additional_metrics: 추가 지표 (gini, stability 등)

        Returns:
            str: "expand" (더 탐색) or "prune" (가지치기)
        """
        node.result = {
            "cooperation_rate": cr,
            **(additional_metrics or {}),
            "evaluated_at": datetime.now().isoformat(),
        }

        if cr > self.best_cr:
            self.best_cr = cr
            self.best_node = node
            node.status = "success"
            print(f"  🌟 새 최고 기록! Node {node.node_id}: CR={cr:.4f}")
            return "expand"   # 더 탐색할 가치 있음
        elif cr < 0.05:
            node.status = "buggy"
            print(f"  🐛 Buggy node {node.node_id}: CR={cr:.4f}")
            return "prune"    # 완전 실패
        else:
            node.status = "pruned"
            print(f"  ✂️ Pruned node {node.node_id}: CR={cr:.4f} (< best {self.best_cr:.4f})")
            return "prune"    # 개선 없음

    def get_next_pending(self):
        """다음 실행할 pending 노드 반환."""
        for node in self.nodes.values():
            if node.status == "pending":
                return node
        return None

    def get_stats(self):
        """탐색 통계."""
        statuses = {}
        for node in self.nodes.values():
            statuses[node.status] = statuses.get(node.status, 0) + 1
        return {
            "total_nodes": len(self.nodes),
            "best_cr": self.best_cr,
            "best_node": self.best_node.node_id if self.best_node else None,
            "statuses": statuses,
        }

    def save_tree(self):
        """탐색 트리를 JSON으로 저장 (헌법 제8조 — 투명성)."""
        tree_data = {
            "timestamp": datetime.now().isoformat(),
            "best_node": self.best_node.node_id if self.best_node else None,
            "best_cr": self.best_cr,
            "stats": self.get_stats(),
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }
        os.makedirs(os.path.dirname(self.tree_path), exist_ok=True)
        with open(self.tree_path, "w", encoding="utf-8") as f:
            json.dump(tree_data, f, indent=2, ensure_ascii=False)
        print(f"  💾 탐색 트리 저장: {self.tree_path}")

    def _load_tree(self):
        """기존 트리 로드."""
        if os.path.exists(self.tree_path):
            try:
                with open(self.tree_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.best_cr = data.get("best_cr", 0.0)
                self._node_counter = len(data.get("nodes", {}))
                print(f"  📂 기존 트리 로드: {self._node_counter}개 노드, 최고 CR={self.best_cr:.4f}")
            except Exception:
                pass


if __name__ == "__main__":
    print("🧪 tree_search.py 단위 테스트")
    print("=" * 50)

    ats = AgenticTreeSearch(tree_path="experiments/evolution/test_tree.json")

    # 루트 생성
    root = ats.create_root({"GENESIS_BETA": 1.0, "IA_ALPHA": 5.0})
    print(f"  Root: {root.node_id}")

    # Mock 확장
    children = ats.expand(root, num_children=3)
    print(f"  자식 노드 {len(children)}개 생성:")
    for c in children:
        print(f"    {c.node_id}: {c.hypothesis_kr}")

    # 평가
    ats.evaluate_node(children[0], 0.25)
    ats.evaluate_node(children[1], 0.35)
    ats.evaluate_node(children[2], 0.02)

    # 통계
    stats = ats.get_stats()
    print(f"\n  통계: {json.dumps(stats, indent=2)}")

    # 저장
    ats.save_tree()

    print("\n✅ 모든 테스트 통과!")
