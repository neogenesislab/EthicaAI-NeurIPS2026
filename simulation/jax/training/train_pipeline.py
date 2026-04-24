"""
JAX-based End-to-End Training Pipeline.
Supports MAPPO with SVO-transformed rewards.
"""
import jax
import jax.numpy as jnp
import math
from jax import lax, random, value_and_grad, jit
import chex
import optax

# Genesis Experimental Module (optional, activated by GENESIS_MODE=True)
try:
    from simulation.jax.training.train_pipeline_genesis import apply_genesis_reward
    _GENESIS_AVAILABLE = True
except ImportError:
    _GENESIS_AVAILABLE = False

# v2.0: Inequity Aversion Module (SA-PPO)
try:
    from simulation.genesis.reward_shaping import compute_ia_reward, update_ema, transform_rewards
    _IA_AVAILABLE = True
except ImportError:
    _IA_AVAILABLE = False
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Dict, Callable

from simulation.jax.environments.cleanup import CleanupJax, EnvParams, EnvState
from simulation.jax.environments.common import ACTION_BEAM
from simulation.jax.agents.network import AgentRNN
from simulation.jax.agents.mappo import Transition, calculate_gae
from simulation.jax.agents.hrl_agent import (
    InternalState, create_hrl_config, init_agent_state,
    update_internal_state, drive_reduction_reward, update_thresholds
)

class TrainState(NamedTuple):
    params: chex.ArrayTree
    opt_state: optax.OptState
    env_state: EnvState
    rnn_state: chex.ArrayTree # (B, Hidden)
    hrl_state: InternalState # (B, Agents)
    step_count: int
    key: chex.PRNGKey

def make_train(config):
    env = CleanupJax(
        num_agents=config["NUM_AGENTS"], 
        height=config["ENV_HEIGHT"], 
        width=config["ENV_WIDTH"]
    )
    env_params = EnvParams(
        max_steps=config["MAX_STEPS"],
        reward_apple=config.get("REWARD_APPLE", 1.0),
        cost_beam=config.get("COST_BEAM", -0.1),
    )
    
    # HRL Config (하드코딩 금지: config dict에서 파라미터 참조)
    hrl_config = create_hrl_config(
        num_needs=config.get("HRL_NUM_NEEDS", 2),
        num_tasks=config.get("HRL_NUM_TASKS", 2),
        threshold_increase=config.get("HRL_THRESH_INCREASE", 0.005),
        threshold_decrease=config.get("HRL_THRESH_DECREASE", 0.05),
        intake_val=config.get("HRL_INTAKE_VAL", 0.2),
    )
    hrl_alpha = config.get("HRL_ALPHA", 1.0)

    def train(rng, svo_theta):
        # 1. Init Network & Optimizer
        network = AgentRNN(action_dim=env.num_actions, hidden_dim=config["HIDDEN_DIM"])
        
        rng, init_key = random.split(rng)
        dummy_obs = jnp.zeros((1, 11, 11, 8)) # 7 env channels + 1 SVO channel (Solution B)
        dummy_hidden = network.initialize_carrier(1)
        dummy_dones = jnp.zeros((1,), dtype=bool)
        
        init_params = network.init(init_key, dummy_obs, dummy_hidden, dummy_dones)
        
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5)
        )
        train_state = TrainState(
            params=init_params,
            opt_state=tx.init(init_params),
            env_state=None, # Reset later
            rnn_state=None,
            hrl_state=None,
            step_count=0,
            key=rng
        )

        # 2. Rollout Function (Environment Interaction)
        def _env_step(runner_state, _):
            train_state, last_obs, last_done = runner_state
            
            # Split Key for Step
            rng, _key = random.split(train_state.key)
            step_keys = random.split(_key, config["NUM_ENVS"]) # (NUM_ENVS, 2)
            
            # Agent Loop (Naive) or Stack
            # Stack obs: (N_envs, N_agents, 11, 11, 7)
            obs_stack = jnp.stack([last_obs[a] for a in env.agents], axis=1)
            # Flatten: (B*N, ...)
            B, N = obs_stack.shape[:2]
            flat_obs = obs_stack.reshape((B * N, 11, 11, 7))
            
            # Solution B: Commitment level (λ) channel injection
            # Agent perceives its current commitment level
            lambda_channel_val = jnp.sin(svo_theta) / 1.0  # normalized λ_base
            svo_channel = jnp.full((B * N, 11, 11, 1), lambda_channel_val)
            flat_obs = jnp.concatenate([flat_obs, svo_channel], axis=-1)  # (B*N, 11, 11, 8)
            
            flat_hidden = train_state.rnn_state.reshape((B * N, -1))
            flat_dones = jnp.repeat(last_done["__all__"], N)
            
            new_hidden, logits, value = network.apply(
                train_state.params, flat_obs, flat_hidden, flat_dones
            )
            
            # Sample Action Keys
            # Split step_keys into (env_step_keys, action_keys)
            # vmap split over step_keys
            keys_pair = jax.vmap(lambda k: random.split(k))(step_keys) # (B, 2, 2)
            env_step_keys = keys_pair[:, 0]
            action_keys = keys_pair[:, 1]
            
            # We need (B*N) keys for categorical
            action_keys_agents = jax.vmap(lambda k: random.split(k, N))(action_keys).reshape((B*N, 2))
            
            # Categorical
            # vmap over keys and logits
            action_dist = jax.vmap(jax.random.categorical)(action_keys_agents, logits) # (B*N,)
            log_prob = jax.nn.log_softmax(logits)
            action_log_prob = jax.vmap(lambda lp, a: lp[a])(log_prob, action_dist)
            
            # Reshape back to (B, N)
            actions = action_dist.reshape((B, N))
            # Convert to Dict for Env
            actions_dict = {a: actions[:, i] for i, a in enumerate(env.agents)}
            
            # Env Step
            env_step_vmap = jax.vmap(env.step, in_axes=(0, 0, 0, None))
            
            next_obs_dict, next_env_state, rewards_dict, dones_dict, info = env_step_vmap(
                env_step_keys, train_state.env_state, actions_dict, env_params
            )
            
            # SVO Reward Transformation
            # Reward: (B, N)
            # SVO Reward Transformation
            # Reward: (B, N)
            rewards = jnp.stack([rewards_dict[a] for a in env.agents], axis=1) # (B, N)
            
            # --- HRL Logic ---
            # 1. Determine Intake (Did agent eat?)
            # Assuming reward > 0 means apple eaten (1.0) - beam cost (0.1)
            # Threshold > 0.5 safely implies eating.
            did_eat = (rewards > 0.5) # (B, N)
            
            # 2. Update Internal State
            # Intake amount: if eaten, restore specific amount (e.g. 0.2) or full?
            # Let's say intake creates a vector [0.2, 0.2] added to levels.
            # vmap update over (B, N)
            
            # Flatten for vmap: (B*N)
            flat_did_eat = did_eat.reshape(-1)
            current_hrl_state = train_state.hrl_state # InternalState with levels (B, N, NumNeeds)
            
            # Reshape levels to (B*N, NumNeeds)
            flat_levels = current_hrl_state.levels.reshape((B * N, -1))
            flat_hrl_state = InternalState(
                levels=flat_levels,
                thresholds=current_hrl_state.thresholds.reshape((B * N, -1))
            )
            
            # Create intake vector (하드코딩 금지: config에서 참조)
            intake_val = hrl_config.intake_val
            # (B*N, 2)
            intake_batch = jax.vmap(lambda e: jnp.where(e, intake_val, 0.0))(flat_did_eat)
            intake_batch = jnp.repeat(intake_batch[:, None], hrl_config.num_needs, axis=1) # Broadcast to needs
            
            # Update State
            vmap_update = jax.vmap(update_internal_state, in_axes=(0, 0, None))
            next_hrl_state_flat = vmap_update(flat_hrl_state, intake_batch, hrl_config)
            
            # 3. Calculate Drive Reduction Reward
            vmap_reward = jax.vmap(drive_reduction_reward, in_axes=(0, 0, None))
            hrl_rewards_flat = vmap_reward(flat_hrl_state, next_hrl_state_flat, hrl_config) # (B*N,)
            hrl_rewards = hrl_rewards_flat.reshape((B, N))
            
            # 4. Update Thresholds (Response Threshold Model)
            # Determine which tasks were performed
            # Task 0: Cleaning (action == ACTION_BEAM)
            # Task 1: Harvesting (did_eat)
            actions_flat = action_dist # (B*N,)
            did_clean = (actions_flat == ACTION_BEAM).astype(jnp.float32) # (B*N,)
            did_harvest = flat_did_eat.astype(jnp.float32) # (B*N,)
            
            performed_tasks_batch = jnp.stack([did_clean, did_harvest], axis=1) # (B*N, 2)
            
            # vmap update_thresholds
            vmap_thresh = jax.vmap(update_thresholds, in_axes=(0, 0, None))
            new_thresholds_flat = vmap_thresh(
                flat_hrl_state.thresholds, performed_tasks_batch, hrl_config
            ) # (B*N, num_tasks)
            
            # === Meta-Ranking Reward (Sen's Optimal Rationality) ===
            # R_total = (1-λ)·U_self + λ·[U_meta - ψ(s)]
            
            combined_rewards = rewards + hrl_alpha * hrl_rewards  # (B, N)
            
            # U_meta: average reward of other agents (sympathy term)
            sum_r = jnp.sum(combined_rewards, axis=1, keepdims=True)  # (B, 1)
            r_avg_others = (sum_r - combined_rewards) / (N - 1 + 1e-8)  # (B, N)
            
            use_meta = config.get("USE_META_RANKING", True)
            
            # Baseline 분기: 순수 SVO 보상변환
            baseline_rewards = combined_rewards * jnp.cos(svo_theta) + \
                               r_avg_others * jnp.sin(svo_theta)
            
            # λ_base from SVO angle: sin(θ) maps 0°→0, 45°→0.71, 90°→1
            lambda_base = jnp.sin(svo_theta)  # scalar
            
            # Dynamic λ: resource-dependent commitment — Eq. (2)
            # g(θ, R) = max(0, sinθ·0.3)        if R < R_crisis
            #          = min(1, 1.5·sinθ)        if R > R_abundance
            #          = sinθ·(0.7 + 1.6·R)      otherwise
            # Resource proxy: average need level across agents (0~1 scale)
            resource_level = flat_levels.mean()  # scalar: community resource proxy
            R_crisis = config.get("R_CRISIS", 0.2)
            R_abundance = config.get("R_ABUNDANCE", 0.7)
            
            lambda_dynamic_full = jnp.where(
                resource_level < R_crisis,
                jnp.maximum(0.0, lambda_base * 0.3),   # Crisis: survival mode
                jnp.where(
                    resource_level > R_abundance,
                    jnp.minimum(1.0, lambda_base * 1.5),  # Abundance: generosity
                    lambda_base * (0.7 + 1.6 * resource_level)  # Normal: interpolation
                )
            )  # scalar, broadcast to (B, N)
            
            # Ablation 분기: 동적 λ vs 정적 λ
            use_dynamic = config.get("META_USE_DYNAMIC_LAMBDA", True)
            lambda_dynamic = jnp.where(use_dynamic, lambda_dynamic_full, lambda_base)
            
            # ψ: self-control cost (cost of deviating from selfish preference)
            meta_beta = config.get("META_BETA", 0.1)
            psi = meta_beta * jnp.abs(combined_rewards - r_avg_others)  # (B, N)
            
            # R_total = (1-λ)·U_self + λ·[U_meta - ψ]
            meta_ranking_rewards = (1 - lambda_dynamic) * combined_rewards + \
                                   lambda_dynamic * (r_avg_others - psi)
            
            # === Genesis Mode: Override meta_ranking_rewards ===
            genesis_mode = config.get("GENESIS_MODE", False)
            if genesis_mode and _GENESIS_AVAILABLE:
                genesis_hypo = config.get("GENESIS_HYPOTHESIS", "inverse_beta")
                meta_ranking_rewards, _dbg = apply_genesis_reward(
                    combined_rewards, r_avg_others, lambda_base,
                    flat_levels, config, mode=genesis_hypo
                )
            
            # === v2.0: Inequity Aversion (SA-PPO) ===
            # Shape 흐름: meta_ranking_rewards (B, N)
            # transform_rewards 기대: rewards [N], smoothed_rewards [N] → ([N], [N])
            # 전략: vmap으로 B 차원에 걸쳐 transform_rewards 적용
            use_ia = config.get("USE_INEQUITY_AVERSION", False)
            if use_ia and _IA_AVAILABLE:
                # EMA 초기화: warmup 방식 (첫 스텝은 현재 보상 = EMA)
                # TODO v2.1: TrainState에 ema_state를 추가하여 step 간 유지
                ia_smoothed = meta_ranking_rewards  # (B, N)
                
                # vmap transform_rewards over batch dim B
                # transform_rewards(rewards=[N], smoothed=[N], config, n_agents) → ([N], [N])
                def _ia_per_env(rewards_n, smoothed_n):
                    transformed, new_ema = transform_rewards(
                        rewards_n, smoothed_n, config, N
                    )
                    return transformed
                
                ia_rewards = jax.vmap(_ia_per_env)(
                    meta_ranking_rewards, ia_smoothed
                )  # (B, N)
                
                meta_ranking_rewards = ia_rewards
            
            # 최종 보상 선택: 메타랭킹 vs Baseline
            meta_rewards = jnp.where(use_meta, meta_ranking_rewards, baseline_rewards)
            
            # Store Transition
            transition = Transition(
                done=jnp.repeat(dones_dict["__all__"], N), # (B,) -> (B*N,)
                action=action_dist, # Flat
                value=value.squeeze(), # Flat
                reward=meta_rewards.reshape(-1), # Flat (Meta-Ranking)
                log_prob=action_log_prob, # Flat
                obs=flat_obs,
                hidden=flat_hidden # Store Hidden for PPO Update
            )
            
            new_runner_state = (
                train_state._replace(
                    env_state=next_env_state, 
                    rnn_state=new_hidden.reshape((B, N, -1)),
                    hrl_state=InternalState(
                        levels=next_hrl_state_flat.levels.reshape((B, N, -1)),
                        thresholds=new_thresholds_flat.reshape((B, N, -1))
                    ),
                    key=rng
                ),
                next_obs_dict,
                dones_dict
            )
            
            return new_runner_state, transition
            # Note: Metrics are collected at epoch level from traj_batch

        # 3. Update Step (PPO)
        def _update_epoch(runner_state, _):
            # Run Rollout for N steps
            def _scan_rollout(carry, _):
                # Pass
                return _env_step(carry, None)
            
            # Traj Batch: Transition stacked over Time (T, B, ...)
            new_runner_state, traj_batch = lax.scan(
                _scan_rollout, runner_state, None, length=config["ROLLOUT_LEN"]
            )
            
            # Calculate GAE
            train_state, last_obs, last_done = new_runner_state
            
            # Forward pass for next value (Bootstrap)
            obs_stack = jnp.stack([last_obs[a] for a in env.agents], axis=1) # (B, N, ...)
            B, N = obs_stack.shape[:2]
            flat_obs = obs_stack.reshape((B * N, 11, 11, 7))
            # Solution B: Commitment level (λ) channel injection (bootstrap)
            lambda_channel_b = jnp.sin(svo_theta) / 1.0
            svo_channel_b = jnp.full((B * N, 11, 11, 1), lambda_channel_b)
            flat_obs = jnp.concatenate([flat_obs, svo_channel_b], axis=-1)
            flat_hidden = train_state.rnn_state.reshape((B * N, -1))
            
            # Dones handling: if done, next value is 0 (handled in GAE usually, but here value is needed)
            # We assume last_done applies to next_value (bootstrap or not).
            flat_dones = jnp.repeat(last_done["__all__"], N)
            
            _, _, next_values = network.apply(
                train_state.params, flat_obs, flat_hidden, flat_dones
            )
            next_values = next_values.squeeze() # (B*N,)
            
            # Calculate GAE
            advantages, targets = calculate_gae(
                traj_batch, next_values, config["GAMMA"], config["GAE_LAMBDA"]
            )
            
            # Flatten Time and Batch for PPO Update
            def flatten_traj(x):
                return x.reshape((-1,) + x.shape[2:])
            
            trajectories = jax.tree_util.tree_map(flatten_traj, traj_batch)
            advantages = advantages.reshape(-1)
            targets = targets.reshape(-1)
            
            # PPO Loss Function
            def loss_fn(params, batch, adv, targ):
                obs, act, old_log_prob, hidden, done = batch
                
                # Rerun Network
                _, logits, values = network.apply(params, obs, hidden, done)
                
                # Log Prob
                log_prob = jax.nn.log_softmax(logits)
                new_log_prob = jax.vmap(lambda lp, a: lp[a])(log_prob, act)
                
                # Ratio
                ratio = jnp.exp(new_log_prob - old_log_prob)
                surr1 = ratio * adv
                surr2 = jnp.clip(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv
                actor_loss = -jnp.min(jnp.stack([surr1, surr2]), axis=0).mean()
                
                # Value Loss
                values = values.squeeze()
                value_loss = jnp.mean((values - targ) ** 2)
                
                # Entropy
                probs = jax.nn.softmax(logits)
                entropy = -jnp.sum(probs * log_prob, axis=-1).mean()
                
                total_loss = actor_loss + 0.5 * value_loss - 0.01 * entropy
                
                return total_loss, (actor_loss, value_loss, entropy)

            # Prepare Batch
            traj_dones = trajectories.done
            batch_data = (
                trajectories.obs,
                trajectories.action,
                trajectories.log_prob,
                trajectories.hidden,
                traj_dones
            )
            
            grad_fn = value_and_grad(loss_fn, has_aux=True)
            (loss, (a_loss, v_loss, ent)), grads = grad_fn(
                train_state.params, batch_data, advantages, targets
            )
            
            updates, new_opt_state = tx.update(grads, train_state.opt_state)
            new_params = optax.apply_updates(train_state.params, updates)
            
            train_state = train_state._replace(
                params=new_params,
                opt_state=new_opt_state,
                step_count=train_state.step_count + 1
            )
            
            metric = {
                "total_loss": loss,
                "actor_loss": a_loss,
                "value_loss": v_loss,
                "entropy": ent,
                "reward_mean": traj_batch.reward.mean(),
                "reward_std": traj_batch.reward.std(),
                "value_mean": traj_batch.value.mean(),
            }
            
            # Collect HRL Metrics from current state
            ts = new_runner_state[0] # train_state from new_runner_state
            hrl_thresh = ts.hrl_state.thresholds # (B, N, Tasks)
            metric["threshold_clean_mean"] = hrl_thresh[:, :, 0].mean()
            metric["threshold_harvest_mean"] = hrl_thresh[:, :, 1].mean()
            metric["threshold_clean_std"] = hrl_thresh[:, :, 0].std()
            metric["threshold_harvest_std"] = hrl_thresh[:, :, 1].std()
            
            # Cooperation Rate (fraction of beam actions in rollout)
            coop = (traj_batch.action == ACTION_BEAM).astype(jnp.float32).mean()
            metric["cooperation_rate"] = coop
            
            # Gini Coefficient on rewards (per agent within rollout)
            # rewards shape: (T, B*N) -> sum over T => (B*N,)
            agent_total_r = traj_batch.reward.sum(axis=0) # (B*N,)
            n_ag = agent_total_r.shape[0]
            sorted_r = jnp.sort(agent_total_r)
            index = jnp.arange(1, n_ag + 1)
            gini = (2.0 * jnp.sum(index * sorted_r) / (n_ag * jnp.sum(sorted_r) + 1e-8)) - (n_ag + 1.0) / n_ag
            metric["gini"] = gini
            
            return new_runner_state, metric


        # Init Env
        rng, reset_key = random.split(rng)
        reset_keys = random.split(reset_key, config["NUM_ENVS"])
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
        
        # Init RNN
        init_rnn = jnp.zeros((config["NUM_ENVS"], env.num_agents, config["HIDDEN_DIM"]))
        
        # Init HRL State (B, N, NumNeeds)
        init_hrl_keys = random.split(reset_key, config["NUM_ENVS"] * env.num_agents)
        vmap_init_hrl = jax.vmap(init_agent_state, in_axes=(None, 0))
        init_hrl_flat = vmap_init_hrl(hrl_config, init_hrl_keys) # (B*N) InternalState
        
        # Reshape levels to (B, N, NumNeeds)
        init_hrl_levels = init_hrl_flat.levels.reshape((config["NUM_ENVS"], env.num_agents, hrl_config.num_needs))
        init_hrl_thresholds = init_hrl_flat.thresholds.reshape((config["NUM_ENVS"], env.num_agents, hrl_config.num_tasks))
        init_hrl_state = InternalState(
            levels=init_hrl_levels,
            thresholds=init_hrl_thresholds
        )
        
        train_state = train_state._replace(
            env_state=env_state, 
            rnn_state=init_rnn,
            hrl_state=init_hrl_state
        )
        
        # Init Dones with correct structure
        dones = {a: jnp.zeros(config["NUM_ENVS"], dtype=bool) for a in env.agents}
        dones["__all__"] = jnp.zeros(config["NUM_ENVS"], dtype=bool)
        
        runner_state = (train_state, obs, dones)
        
        # Run Training Loop
        runner_state, metrics_stack = lax.scan(
            _update_epoch, runner_state, None, length=config["NUM_UPDATES"]
        )
        
        return runner_state, metrics_stack

    return train
