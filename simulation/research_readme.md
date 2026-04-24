# EthicaAI: Follow-up Research (Post-NeurIPS)

## 1. LLM Agent Prototype (Constitutional AI)
- **Path**: `simulation/llm/llm_agent_prototype.py`
- **Description**: Implements Sen's Meta-Ranking using Constitutional AI principles. The agent switches between "Self-Interest" and "Social Welfare" modes based on resource levels defined in the system prompt.
- **Usage**: `python simulation/llm/llm_agent_prototype.py`

## 2. Human-AI Comparison
- **Path**: `simulation/jax/analysis/human_ai_comparison.py`
- **Description**: Compares the distribution of cooperation rates and Gini coefficients between EthicaAI agents and human Public Goods Game datasets (Zenodo/OSF).
- **Metric**: Wasserstein Distance.

## 3. Baseline Comparison
- **Path**: `simulation/jax/analysis/baseline_comparison.py`
- **Description**: Statistical significance test (t-test, Cohen's d) between Full Model (Meta-Ranking ON) and Baseline (OFF).

## Next Steps
1. Integrate actual LLM API (OpenAI/Gemini).
2. Download real human PGG data CSV.
3. Conduct evolutionary competition simulation.
