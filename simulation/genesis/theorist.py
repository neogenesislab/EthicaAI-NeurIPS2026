import os
import json
import random
try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    genai = None
    _GENAI_AVAILABLE = False
from simulation.jax.config import get_config

class Theorist:
    def __init__(self, history_path="experiments/evolution/history.json"):
        self.history_path = history_path
        
        # Load .env manually to avoid dependency
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("GEMINI_API_KEY="):
                        os.environ["GEMINI_API_KEY"] = line.strip().split("=", 1)[1]
                        break
        
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if self.api_key and _GENAI_AVAILABLE:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        else:
            if not _GENAI_AVAILABLE:
                print("⚠️ google.generativeai 미설치. MOCK MODE로 진행합니다.")
            else:
                print("⚠️ GEMINI_API_KEY not found. Running in MOCK MODE.")
            self.model = None

    def load_history(self):
        if os.path.exists(self.history_path):
            with open(self.history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def propose_next_config(self):
        history = self.load_history()
        
        # 1. Base Config
        base_config = get_config("medium")
        
        # 2. Generate Proposal
        if self.model:
            proposal = self._query_llm(history)
        else:
            proposal = self._mock_proposal(history)
            
        print(f"🧠 Theorist proposed: {json.dumps(proposal, indent=2)}")
        
        # 3. Merge with Base Config
        # v2.0: Genesis + IA 파라미터 모두 적용
        next_config = base_config.copy()
        for key, value in proposal.items():
            if key.startswith("GENESIS_") or key.startswith("IA_") or key == "USE_INEQUITY_AVERSION":
                next_config[key] = value
                
        return next_config, proposal

    def _mock_proposal(self, history):
        # v2.0: Random Search (Genesis + IA 파라미터)
        return {
            "GENESIS_BETA": round(random.uniform(0.1, 10.0), 2),
            "GENESIS_ALPHA": round(random.uniform(0.1, 5.0), 2),
            "GENESIS_LOGIC_MODE": random.choice(["adaptive_beta", "inverse_beta", "institutional"]),
            "IA_ALPHA": round(random.uniform(0.5, 10.0), 2),
            "IA_BETA": round(random.uniform(0.01, 1.0), 3),
            "USE_INEQUITY_AVERSION": random.choice([True, False]),
            "rationale": "Mock Theorist: Random exploration (Genesis + IA params)."
        }

    def _query_llm(self, history):
        prompt = f"""
        You are an AI Scientist optimizing a Multi-Agent System (EthicaAI).
        Your goal is to maximize the Cooperative Rate (0.0 to 1.0) in a Social Dilemma.
        
        Current Status:
        - Previous Experiments: {json.dumps(history[-5:], indent=2)}
        
        Task:
        - Analyze the history.
        - Propose the NEXT set of hyperparameters to test.
        - Focus on the following parameters:
          * `GENESIS_BETA` (intervention strength, 0.01~100)
          * `GENESIS_ALPHA` (sensitivity, 0.01~5)
          * `GENESIS_LOGIC_MODE` (one of: "adaptive_beta", "inverse_beta", "institutional")
          * `IA_ALPHA` (envy coefficient, 0.1~10.0) [v2.0: Inequity Aversion]
          * `IA_BETA` (guilt coefficient, 0.01~1.0)  [v2.0: Inequity Aversion]
          * `USE_INEQUITY_AVERSION` (true/false)      [v2.0: SA-PPO 활성화]
        
        Output JSON Format:
        {{
            "GENESIS_BETA": float,
            "GENESIS_ALPHA": float,
            "GENESIS_LOGIC_MODE": "adaptive_beta" | "inverse_beta" | "institutional",
            "IA_ALPHA": float,
            "IA_BETA": float,
            "USE_INEQUITY_AVERSION": bool,
            "rationale": "One sentence explaining why in English.",
            "rationale_kr": "위 내용을 한국어로 번역한 한 문장."
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Simple cleanup for JSON parsing
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            print(f"❌ LLM Query Failed: {e}")
            return self._mock_proposal(history)

if __name__ == "__main__":
    theorist = Theorist()
    config, proposal = theorist.propose_next_config()
    
    # Save for next step
    os.makedirs("experiments/evolution/current", exist_ok=True)
    with open("experiments/evolution/current/config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    with open("experiments/evolution/current/hypothesis.md", "w") as f:
        f.write(f"# Hypothesis\n\n{proposal.get('rationale', 'No rationale')}")
