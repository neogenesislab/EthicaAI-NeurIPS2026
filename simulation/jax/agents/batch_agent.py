"""
JAX-based Batch Agent
Vectorized decision making for 10,000+ agents.
"""
import jax
import jax.numpy as jnp
from jax import random
from flax import struct
import chex
from typing import Tuple

@struct.dataclass
class AgentParams:
    # Meta-Ranking Parameters
    commitment_lambda: float # 0.0 ~ 1.0 (Scalar per agent)
    # Self-control cost for cooperation when selfish utility favors defection
    psi: float = 0.1 

@struct.dataclass
class AgentState:
    wealth: float = 100.0
    # Add mor state if needed (e.g. reputation)

class BatchAgent:
    """
    Represents a population of agents.
    Designed to work with vmap.
    """
    
    @staticmethod
    def act(key: chex.PRNGKey, obs: chex.Array, params: AgentParams) -> int:
        """
        Decide action for a SINGLE agent.
        Will be vmapped later.
        
        obs: [Opp_C, Opp_D, Opp_P, Init]
        """
        # 1. Parse Observation
        # obs is float array
        opp_c = obs[0] > 0.5
        opp_d = obs[1] > 0.5
        opp_p = obs[2] > 0.5
        is_init = obs[3] > 0.5
        
        # 2. Calculate Utilities for 4 actions
        # 0:C, 1:D, 2:C+P, 3:D+P
        
        # We need a Utility Model similar to OptimalAgent
        # U_total = (1-lambda)*U_self + lambda*U_social - Cost_sc
        
        # Simplified Payoff Expectation (Assumes Opponent repeats last action)
        # If Init, assume Cooperate (Optimistic) or Random? Let's say Random or C.
        
        # Payoff Matrix (Standard Params from Config, hardcoded here for JIT simplicity or pass as args)
        T, R, P, S = 5.0, 3.0, 1.0, 0.0
        PunishCost = 1.0
        PunishFine = 4.0
        
        # Predict Opponent Action
        # Limit: In PD, prediction is key. Here we use "Opponent stays" heuristic (Tit-for-Tatish belief)
        # Or Just calculate utility against C and D and weight by probability?
        
        # Let's assume Opponent will do what they did last time (Deterministic Belief)
        # If init, assume C
        
        pred_opp_c = jnp.where(is_init, 1.0, jnp.where(opp_c, 1.0, 0.0))
        pred_opp_d = jnp.where(is_init, 0.0, jnp.where(opp_d, 1.0, 0.0))
        # Opponent Punish? Assuming opponent keeps punishing if I defect? 
        # For simplicity, ignore opponent's future punishment dynamics in utility calc for now, 
        # or assume opponent punishes if I defect (Social Norm).
        
        # --- Utility Calculation ---
        
        # Case 1: I choose C (0)
        # Payoff: R if Opp_C, S if Opp_D
        u_self_c = pred_opp_c * R + pred_opp_d * S
        u_soc_c = pred_opp_c * (R + R) + pred_opp_d * (S + T) # Joint Utility
        
        # Case 2: I choose D (1)
        # Payoff: T if Opp_C, P if Opp_D
        # BUT if Opponent Punishes? (Opp_P is active) -> I get Fine
        # Or if Social Norm exists?
        
        # Refined: If Opponent Punished last time, they might punish again?
        # Let's stick to basic PD payoffs first.
        u_self_d = pred_opp_c * T + pred_opp_d * P
        u_soc_d = pred_opp_c * (T + S) + pred_opp_d * (P + P)
        
        # Case 3: C+P (2)
        # Payoff: Same as C, but minus PunishCost
        # Value: Payoff is lower, BUT meaningful if we value "Teaching" or "Norm"
        # In current utility function, Punishment is only chosen if it increases Long-term utility
        # or if Lambda is high enough to value "Correcting" the opponent.
        # Capturing "Altruistic Punishment" in single-step utility is hard without stateful belief.
        
        # HACK: Add "Norm Utility" to P actions if Opponent Defected
        # If Opponent Defected, Punishing yields +Alpha social utility (Justice)
        norm_bonus = jnp.where(opp_d, 5.0, 0.0) # Justice Bonus
        
        u_self_cp = u_self_c - PunishCost
        u_soc_cp = u_soc_c - PunishCost - PunishFine # I pay cost, Opp pays fine (Socially destructive?)
        # Actually Social Welfare = (R-Cost) + (R-Fine)? No.
        # Sympathy means I care about Opponent. Punishing opponent Lowers their payoff.
        # So classic Sympathy (Beta) actually DISCOURAGES punishment.
        # We need "Gamma" (Commitment to Norm).
        
        # Implementing the full Meta-Ranking Utility:
        # U = (1-lambda)*U_self + lambda*U_norm
        
        # U_norm:
        #  - Cooperate against Cooperator: High
        #  - Defect against Cooperator: Low
        #  - Punish Defector: High
        
        # Simple Proxy for Norm Utility:
        # C is good. D is bad.
        # P is good ONLY if Opponent Defected.
        
        val_c = 1.0
        val_d = -1.0
        val_p = jnp.where(opp_d, 2.0, -2.0) # Punishing defector is Very Good, punishing cooperator is Very Bad
        
        u_norm_c = val_c
        u_norm_d = val_d
        u_norm_cp = val_c + val_p
        u_norm_dp = val_d + val_p
        
        # Combine
        # Standard Agent: lambda * U_norm + (1-lambda) * U_self
        # Note: U_self needs to be normalized or U_norm scaled.
        # Payoffs are around 0~5. Norm values 1~2. Scale Norm by 2.0?
        
        lam = params.commitment_lambda
        
        total_c  = (1 - lam) * u_self_c  + lam * (u_norm_c * 2.0)
        total_d  = (1 - lam) * u_self_d  + lam * (u_norm_d * 2.0)
        total_cp = (1 - lam) * u_self_cp + lam * (u_norm_cp * 2.0)
        total_dp = (1 - lam) * u_self_dp + lam * (u_norm_dp * 2.0)
        
        # Self-Control Cost (Psi)
        # If Selfish preference is D, but we choose C, pay Psi
        # Here we just select Max U.
        
        # Stack Utilities
        utilities = jnp.array([total_c, total_d, total_cp, total_dp])
        
        # Action Selection (Deterministic greedy)
        best_action = jnp.argmax(utilities)
        
        # Optional: Add noise (Bounded Rationality) via Gumbel-Max trick if key provided?
        # For now, deterministic.
        
        return best_action

    @staticmethod
    def batch_act(key: chex.PRNGKey, obs_batch: chex.Array, params_batch: AgentParams) -> chex.Array:
        # vmap over batch dimension
        # obs_batch: (B, 4)
        # params_batch: AgentParams with fields (B,)
        
        # We need to map over structured params
        # Use jax.vmap
        
        return jax.vmap(BatchAgent.act, in_axes=(None, 0, 0))(key, obs_batch, params_batch)
