import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
th.backends.cuda.preferred_linalg_library("magma")

# from stable_baselines3.common.buffers import RolloutBuffer
# from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance

from mask_pack.common.buffers import RolloutBuffer
from mask_pack.common.on_policy_algorithm import OnPolicyAlgorithm
from mask_pack.common.policies import CustomActorCriticPolicy
from mask_pack.common.utils import get_schedule_fn
from mask_pack.acktr.kfac import KFACOptimizer

SelfACKTR = TypeVar("SelfACKTR", bound="ACKTR")


class ACKTR(OnPolicyAlgorithm):
    """
    ACKTR
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "CnnMlpPolicy": CustomActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[CustomActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        # batch_size: int = 64,
        # n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        # normalize_advantage: bool = True,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        # target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,

        acktr = True,
        eps = None,
        alpha = None,
        add_mask_loss: bool = True,
        mask_coef: float = 0.5,
        add_entropy_loss: bool = True,
        ent_coef: float = 0.0,
        add_invalid_probs: bool = True,
        invalid_probs_coef: float = 0.0,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        self.acktr = acktr
        self.eps = eps
        self.alpha = alpha
        self.loss_func = th.nn.MSELoss(reduce=False, size_average=True)
        self.add_mask_loss = add_mask_loss
        self.mask_coef = mask_coef
        self.add_entropy_loss = add_entropy_loss
        self.add_invalid_probs = add_invalid_probs
        self.invalid_probs_coef = invalid_probs_coef

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if self.acktr:
            self.policy.optimizer = KFACOptimizer(self.policy)
        else:
            self.policy.optimizer = th.optim.RMSprop(self.policy.parameters(), lr=self.lr_schedule(1), eps=self.eps, alpha=self.alpha)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        for r in self.rollout_buffer.get(self.n_steps * self.env.num_envs):
            rollout_data = r

        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = rollout_data.actions.long().flatten()

        values, log_prob, entropy, invalid_probs = self.policy.evaluate_actions(rollout_data.observations, actions)

        values = values.view(self.n_steps, self.env.num_envs, 1)        
        log_prob = log_prob.view(self.n_steps, self.env.num_envs, 1)    
        
        advantages = rollout_data.returns.view(self.n_steps, self.env.num_envs, 1) - values
        value_loss = advantages.pow(2).mean() 
        action_loss = -(advantages.detach() * log_prob).mean()
        
        mask_len = self.action_space.n
        pred_mask = self.policy.predict_masks(rollout_data.observations).reshape((self.n_steps, self.env.num_envs, mask_len))
        truth_mask = rollout_data.truth_masks.reshape(self.n_steps, self.env.num_envs, mask_len)   
        mask_loss = self.loss_func(pred_mask, truth_mask).mean()
        
        entropy_loss = entropy.mean() 
        invalid_probs_loss = invalid_probs.mean()  

        if self.acktr and self.policy.optimizer.steps % self.policy.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.policy.zero_grad()
            pg_fisher_loss = -log_prob.mean()

            value_noise = th.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean() # detach

            fisher_loss = pg_fisher_loss + vf_fisher_loss + mask_loss * 1e-8
            # fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.policy.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.policy.optimizer.acc_stats = False

        self.policy.optimizer.zero_grad()
        loss = value_loss * self.vf_coef
        loss += action_loss
        loss += invalid_probs_loss * self.invalid_probs_coef
        loss -= entropy_loss * self.ent_coef
        loss += mask_loss * self.mask_coef
        loss.backward()

        if self.acktr == False:
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

        self.policy.optimizer.step()

        self._n_updates += 1

        # Logs
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/policy_gradient_loss", action_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/mask_loss", mask_loss.item())
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/invalid_probs_loss", invalid_probs_loss.item())
        self.logger.record("train/n_updates", self._n_updates)


    def learn(
        self: SelfACKTR,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "ACKTR",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfACKTR:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
