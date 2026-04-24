import os
import json
import numpy as np

class Coordinator:
    """
    The Coordinator (Meta-Agent)
    Monitors the evolution process and intervenes when stagnation involves.
    """
    def __init__(self, history_path="experiments/evolution/history.json"):
        self.history_path = history_path
        self.stagnation_threshold = 0.0001
        self.patience = 5  # Generations to wait before intervening

    def check_stagnation(self):
        if not os.path.exists(self.history_path):
            return False, "No history found."
        
        with open(self.history_path, "r") as f:
            history = json.load(f)
            
        if len(history) < self.patience:
            return False, "Not enough history."
            
        # Extract cooperation rates from last N generations
        recent_coops = []
        for item in history[-self.patience:]:
            coop = item.get("result", {}).get("Prosocial", {}).get("cooperation_rate", 0.0)
            recent_coops.append(coop)
            
        # Calculate Variance
        variance = np.var(recent_coops)
        
        if variance < self.stagnation_threshold:
            return True, f"Stagnation Detected (Var={variance:.6f} < {self.stagnation_threshold})"
        
        return False, f"Progressing (Var={variance:.6f})"

    def intervene(self, method="reset"):
        """
        v2.0: 3단계 개입 시스템.
        - Level 1 (Poke): IA 파라미터 소폭 조정
        - Level 2 (Shock): 환경 변수 극적 변경
        - Level 3 (Reset): 히스토리 완전 초기화
        """
        print(f"🚨 Coordinator Intervening: {method}")
        
        if method == "poke":
            # Level 1: IA 파라미터 소폭 조정
            config_path = "experiments/evolution/current/config.json"
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                # 질투 계수를 20% 증가시켜 배신자 처벌 강화
                config["IA_ALPHA"] = config.get("IA_ALPHA", 5.0) * 1.2
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)
                print(f"  > Poke: IA_ALPHA → {config['IA_ALPHA']:.2f}")
            
        elif method == "shock":
            # Level 2: 환경 변수 극적 변경
            config_path = "experiments/evolution/current/config.json"
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                # Genesis 로직 모드를 강제 전환
                modes = ["adaptive_beta", "inverse_beta", "institutional"]
                current = config.get("GENESIS_LOGIC_MODE", "adaptive_beta")
                new_mode = [m for m in modes if m != current][0]
                config["GENESIS_LOGIC_MODE"] = new_mode
                config["GENESIS_BETA"] = config.get("GENESIS_BETA", 1.0) * 5.0
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)
                print(f"  > Shock: Mode {current} → {new_mode}, Beta x5")
            
        elif method == "reset":
            # Level 3: Hard Reset
            if os.path.exists(self.history_path):
                os.remove(self.history_path)
            print("  > History wiped. Theorist brainwashed.")
            
if __name__ == "__main__":
    coordinator = Coordinator()
    is_stagnant, message = coordinator.check_stagnation()
    print(f"Coordinator Status: {message}")
    
    if is_stagnant:
        coordinator.intervene(method="reset")
