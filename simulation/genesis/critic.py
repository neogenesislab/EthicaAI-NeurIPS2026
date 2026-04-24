import os
import json
import pandas as pd
from datetime import datetime

class Critic:
    def __init__(self):
        self.result_path = "experiments/evolution/current/result.json"
        self.config_path = "experiments/evolution/current/config.json"
        self.history_path = "experiments/evolution/history.json"
        self.report_path = "experiments/evolution/current/report.md"

    def evaluate(self, success_criteria=None):
        # Load Data
        with open(self.result_path, "r") as f:
            results = json.load(f)
        with open(self.config_path, "r") as f:
            config = json.load(f)

        # Analysis
        prosocial = results.get("Prosocial", {})
        individualist = results.get("Individualist", {})
        prosocial_coop = prosocial.get("cooperation_rate", 0.0)
        individualist_coop = individualist.get("cooperation_rate", 0.0)
        
        # 동적 성공 기준 (Research Director에서 받거나, 기본값 0.5)
        if success_criteria:
            target = success_criteria.get("target", 0.5)
        else:
            target = 0.5
        success = prosocial_coop > target
        
        # v2.0: 다차원 지표 계산
        coop_std = prosocial.get("cooperation_std", 0.0)
        gini = prosocial.get("gini", 0.0)
        num_seeds = prosocial.get("num_seeds", 1)
        platform = prosocial.get("platform", "cpu")

        # 안정성 지수(S): 분산이 낮을수록 안정적 (헌법 제12조 3항)
        max_variance = 0.25  # 이론적 최대 분산 (0~1 범위의 지표)
        stability_index = 1.0 - min(coop_std ** 2 / max_variance, 1.0)
        
        # Update History
        history_item = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "result": results,
            "success": success,
            "stability_index": stability_index,
            "platform": platform,
        }
        self._update_history(history_item)
        
        # Generate Report (v2.0 확장)
        report = f"""# 🧐 Critic Report (v2.0)
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Status**: {"✅ SUCCESS" if success else "❌ FAILURE"}
**Platform**: {platform.upper()}

## 1. Parameters
- **Beta**: {config.get("GENESIS_BETA")}
- **Alpha**: {config.get("GENESIS_ALPHA")}
- **Mode**: {config.get("GENESIS_LOGIC_MODE")}
- **IA Active**: {config.get("USE_INEQUITY_AVERSION", False)}
- **IA Alpha (Envy)**: {config.get("IA_ALPHA", "N/A")}
- **IA Beta (Guilt)**: {config.get("IA_BETA", "N/A")}

## 2. Results
- **Prosocial Coop**: {prosocial_coop:.4f} (Target: > {target})
- **Individualist Coop**: {individualist_coop:.4f}

## 3. 다차원 지표 (v2.0)
- **안정성 지수(S)**: {stability_index:.4f}
- **Coop 표준편차**: {coop_std:.4f} (시드 {num_seeds}개)
- **Gini 계수**: {gini:.4f}
- **시드별 결과**: {prosocial.get("cooperation_per_seed", "N/A")}

## 4. Verdict
{"The experiment succeeded! We found the optimal parameters." if success else "The experiment failed. The parameters did not induce sufficient cooperation."}
"""
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(report)
            
        print(f"🧐 Critic: Report generated. Success={success} | S={stability_index:.4f} | CR={prosocial_coop:.4f}±{coop_std:.4f}")
        return success

    def _update_history(self, item):
        history = []
        if os.path.exists(self.history_path):
            with open(self.history_path, "r") as f:
                history = json.load(f)
        
        history.append(item)
        
        # Save History
        with open(self.history_path, "w") as f:
            json.dump(history, f, indent=4)

if __name__ == "__main__":
    critic = Critic()
    critic.evaluate()
