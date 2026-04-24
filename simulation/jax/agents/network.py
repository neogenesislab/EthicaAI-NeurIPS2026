"""
Flax-based Agent Network (CNN-GRU)
Implements Actor-Critic architecture for POMDP.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
import chex
from typing import Tuple, Sequence

class EncodeCNN(nn.Module):
    """
    CNN Encoder for 11x11 Grid Observation.
    """
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        # x: (B, 11, 11, 6)
        
        # Layer 1
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1), padding='VALID')(x)
        x = nn.relu(x)
        
        # Layer 2
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='VALID')(x)
        x = nn.relu(x)
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        # Dense
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        
        return x

class AgentRNN(nn.Module):
    """
    Actor-Critic RNN Agent.
    """
    action_dim: int = 8
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs, hidden, dones):
        """
        obs: (B, 11, 11, 6)
        hidden: (B, hidden_dim) - GRU hidden state
        dones: (B,) - Reset state if done
        """
        
        # 1. Encode Observation
        # CNN applied to obs
        embedding = EncodeCNN(hidden_dim=self.hidden_dim)(obs)
        
        # 2. Recurrent Layer (GRU)
        # Flax GRUCell expects (carry, inputs)
        # We need to handle 'dones' to reset hidden state.
        
        # Reset hidden if done
        # hidden = jnp.where(dones[:, None], jnp.zeros_like(hidden), hidden)
        # Or handled outside? Usually handled inside model or training loop.
        # Let's handle masking outside commonly, or here.
        
        rnn_input = embedding
        
        gru = nn.GRUCell(features=self.hidden_dim)
        
        new_hidden, rnn_out = gru(hidden, rnn_input)
        
        # 3. Heads
        
        # Actor Head
        actor_logits = nn.Dense(features=self.action_dim)(rnn_out)
        
        # Critic Head
        # IPPO: Value depends on local observation history (rnn_out)
        # MAPPO: Value depends on Global State.
        # For now, implement IPPO style (Value from local history).
        # To support MAPPO, we need a separate Value Network accepting Global State.
        
        value = nn.Dense(features=1)(rnn_out)
        
        return new_hidden, actor_logits, value

    def initialize_carrier(self, batch_size):
        return jnp.zeros((batch_size, self.hidden_dim))

class CriticNetwork(nn.Module):
    """
    Centralized Critic for MAPPO.
    Takes Global State (or concatenated obs).
    """
    hidden_dim: int = 128
    
    @nn.compact
    def __call__(self, global_state):
        # global_state: (B, State_Dim) or (B, H, W, C)
        # If grid, use CNN. If vector, use MLP.
        pass
