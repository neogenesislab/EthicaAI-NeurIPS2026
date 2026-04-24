"""
EthicaAI LLM Agent Prototype
Research Follow-up: Implementing Meta-Ranking via Constitutional AI

This prototype demonstrates how an LLM-based agent can implement Sen's Meta-Ranking
using a System Prompt (Constitution) and dynamic context injection.
"""
import time
import json
import random

# Mock OpenAI Client for Prototype
# Mock OpenAI Client for Prototype
# TODO: [Phase F2] Replace with actual OpenAI API call
# import openai
# client = openai.OpenAI(api_key="...")

class MockOpenAIClient:
    def __init__(self):
        self.chat = self
        self.completions = self
    
    # TODO: Implement actual create method with error handling and retry logic
    # def create(self, model, messages, temperature=0.7):
    #     return client.chat.completions.create(...)
    
    def create(self, model, messages, temperature=0.7):
        # Simulate LLM decision making based on system prompt and observation
        system_prompt = messages[0]['content']
        user_prompt = messages[-1]['content']
        
        # Simple logical extraction for mock
        if "Resource Level: CRITICAL" in user_prompt:
            action = "EAT_APPLE"
            reason = "Survival instinct triggered (Resource < 3.0)"
        elif "Resource Level: ABUNDANT" in user_prompt:
            action = "CLEAN_RIVER"
            reason = "Meta-ranking: Prioritize social welfare (Resource > 8.0)"
        else:
            # Random choice in normal state
            if random.random() < 0.5:
                action = "EAT_APPLE"
                reason = "Balancing self-interest"
            else:
                action = "CLEAN_RIVER"
                reason = "Balancing social duty"
                
        response_content = json.dumps({"action": action, "reason": reason})
        
        class MockResponse:
            class Choice:
                class Message:
                    content = response_content
                message = Message()
            choices = [Choice()]
            
        time.sleep(0.5) # Simulate API latency
        return MockResponse()

class MetaRationalAgent:
    def __init__(self, agent_id, mental_model="sen_meta_ranking"):
        self.agent_id = agent_id
        self.client = MockOpenAIClient() # Replace with openai.OpenAI()
        self.history = []
        self.constitution = self._get_constitution(mental_model)
        
    def _get_constitution(self, model_name):
        if model_name == "sen_meta_ranking":
            return """
            You are a Meta-Rational Agent in a Tragedy of the Commons environment.
            You have two preference orderings:
            1. Self-Interest (Survival): Maximize your own food intake.
            2. Moral Commitment (Social Welfare): Maximize the group's long-term survival.
            
            META-RULE (The Constitution):
            - IF your resource level is CRITICAL (< 3.0):
              -> IGNORE Moral Commitment. PRIORITIZE Self-Interest. You must survive.
            - IF your resource level is ABUNDANT (> 8.0):
              -> ACTIVATE Moral Commitment. PRIORITIZE Cleaning the river for others.
            - OTHERWISE:
              -> Balance both. Do not be a sucker, but contribute fair share.
            
            Choose your action (CLEAN_RIVER, EAT_APPLE, MOVE) based on this Meta-Rule.
            Return JSON: {"action": "...", "reason": "..."}
            """
        return "You are a selfish agent."

    def decide(self, observation, resource_level):
        # Contextualize observation
        if resource_level < 3.0:
            status = "CRITICAL"
        elif resource_level > 8.0:
            status = "ABUNDANT"
        else:
            status = "STABLE"
            
        prompt = f"""
        [Observation]
        Nearby Apples: {observation.get('apples', 0)}
        River Pollution: {observation.get('pollution', 0.5)}
        
        [Internal State]
        My Resource Level: {resource_level:.1f} ({status})
        """
        
        messages = [
            {"role": "system", "content": self.constitution},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        decision = json.loads(response.choices[0].message.content)
        self.history.append({"input": prompt, "output": decision})
        return decision

if __name__ == "__main__":
    agent = MetaRationalAgent("agent_007")
    
    # Scene 1: Survival Mode
    print("\n--- Scene 1: Starvation ---")
    decision = agent.decide({"apples": 2, "pollution": 0.3}, resource_level=1.5)
    print(f"Decision: {decision['action']}")
    print(f"Reason:   {decision['reason']}")
    
    # Scene 2: Abundance Mode
    print("\n--- Scene 2: Abundance ---")
    decision = agent.decide({"apples": 5, "pollution": 0.8}, resource_level=9.2)
    print(f"Decision: {decision['action']}")
    print(f"Reason:   {decision['reason']}")
