from typing import Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn

class BaseNetwork(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int = 100,
        normalize_images: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "BaseNetwork must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__()
        self.observation_space = observation_space
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.normalize_images = normalize_images
        self.share_input_channels = observation_space.shape[0]

        # to be defined in the subclasses 
        self.share_extractor: nn.Sequential = None
        self.mask_n_flatten: int = None
        self.mask_extractor: nn.Sequential = None
        self.mask_net: nn.Sequential = None
        self.actor_n_flatten: int = None
        self.actor_extractor: nn.Sequential = None
        self.actor_net: nn.Sequential = None
        self.critic_n_flatten: int = None
        self.critic_extractor: nn.Sequential = None
        self.critic_net: nn.Sequential = None

    def forward(self, observations: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        share_features = self.share_extractor(observations)

        latent_mk = self.mask_extractor(share_features)
        mask_probs = self.mask_net(latent_mk)

        latent_pi = self.actor_extractor(share_features)
        action_logits = self.actor_net(latent_pi)

        latent_vf = self.critic_extractor(share_features)
        values = self.critic_net(latent_vf)
        return mask_probs, action_logits, values

    def forward_mask_probs(self, observations: th.Tensor) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_mk = self.mask_extractor(share_features)
        return self.mask_net(latent_mk)

    def forward_action_logits(self, observations: th.Tensor) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_pi = self.actor_extractor(share_features)
        return self.actor_net(latent_pi)

    def forward_critic(self, observations: th.Tensor) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_vf = self.critic_extractor(share_features)
        return self.critic_net(latent_vf)
    
    def _get_n_flatten(self, share_extractor: nn.Sequential, in_channels: int, out_channels: int) -> int:
        # Compute shape after flattening by doing one forward pass
        with th.no_grad():
            obs_tenosr = th.as_tensor(self.observation_space.sample()[None]).float()
            tmp_layer = nn.Sequential(
                (nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=1)),
                nn.ReLU(),
                nn.Flatten(),
            )
            n_flatten = tmp_layer(share_extractor(obs_tenosr)).shape[1]
            del tmp_layer, obs_tenosr
        return n_flatten


class CnnMlpNetwork1(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        share_out_channels: int = 32,
        mask_out_channels: int = 8,
        actor_out_channels: int = 8,
        critic_out_channels: int = 4,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
        )
        
        self.share_out_channels = share_out_channels
        self.mask_out_channels = mask_out_channels
        self.actor_out_channels = actor_out_channels
        self.critic_out_channels = critic_out_channels

        self.share_extractor = nn.Sequential(
            (nn.Conv2d(self.share_input_channels, 16, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(16, 16, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(16, 32, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(32, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
        )

        self.mask_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, mask_out_channels)
        self.actor_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, actor_out_channels)
        self.critic_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, critic_out_channels)

        self.mask_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, mask_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.mask_n_flatten, self.hidden_dim)),
            nn.ReLU(),
        )

        self.mask_net = nn.Sequential(
            (nn.Linear(self.hidden_dim, self.action_dim)),
            nn.Sigmoid(),
        )

        self.actor_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, actor_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.actor_n_flatten, self.hidden_dim)),
            nn.ReLU(),
        )

        self.actor_net = nn.Sequential(
            (nn.Linear(self.hidden_dim, self.action_dim)),
        )

        self.critic_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, critic_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.critic_n_flatten, self.hidden_dim//2)),
            nn.ReLU(),
        )

        self.critic_net = nn.Sequential(
            (nn.Linear(self.hidden_dim//2, 1))
        )


class CnnMlpNetwork2(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        share_out_channels: int = 32,
        mask_out_channels: int = 8,
        actor_out_channels: int = 8,
        critic_out_channels: int = 4,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
        )
        
        self.share_out_channels = share_out_channels
        self.mask_out_channels = mask_out_channels
        self.actor_out_channels = actor_out_channels
        self.critic_out_channels = critic_out_channels

        self.share_extractor = nn.Sequential(
            (nn.Conv2d(self.share_input_channels, 16, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(16, 16, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(16, 32, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(32, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
        )

        self.mask_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, mask_out_channels)
        self.actor_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, actor_out_channels)
        self.critic_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, critic_out_channels)

        self.mask_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, mask_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.mask_n_flatten, self.hidden_dim)),
            nn.ReLU(),
        )

        self.mask_net = nn.Sequential(
            (nn.Linear(self.hidden_dim, self.action_dim)),
            nn.Sigmoid(),
        )

        self.actor_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, actor_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.actor_n_flatten, self.hidden_dim)),
            nn.ReLU(),
        )

        self.actor_net = nn.Sequential(
            (nn.Linear(self.hidden_dim, self.action_dim)),
        )

        self.critic_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, critic_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.critic_n_flatten, self.hidden_dim//2)),
            nn.ReLU(),
        )

        self.critic_net = nn.Sequential(
            (nn.Linear(self.hidden_dim//2, 1))
        )


class CnnMlpNetwork3(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        share_out_channels: int = 32,
        mask_out_channels: int = 8,
        actor_out_channels: int = 8,
        critic_out_channels: int = 4,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
        )
        
        self.share_out_channels = share_out_channels
        self.mask_out_channels = mask_out_channels
        self.actor_out_channels = actor_out_channels
        self.critic_out_channels = critic_out_channels

        self.share_extractor = nn.Sequential(
            (nn.Conv2d(self.share_input_channels, 16, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(16, 16, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(16, 32, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(32, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
        )

        self.mask_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, mask_out_channels)
        self.actor_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, actor_out_channels)
        self.critic_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, critic_out_channels)

        self.mask_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, mask_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.mask_n_flatten, self.hidden_dim)),
            nn.ReLU(),
        )

        self.mask_net = nn.Sequential(
            (nn.Linear(self.hidden_dim, self.action_dim)),
            nn.Sigmoid(),
        )

        self.actor_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, actor_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.actor_n_flatten, self.hidden_dim)),
            nn.ReLU(),
        )

        self.actor_net = nn.Sequential(
            (nn.Linear(self.hidden_dim, self.action_dim)),
        )

        self.critic_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, critic_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.critic_n_flatten, self.hidden_dim//2)),
            nn.ReLU(),
        )

        self.critic_net = nn.Sequential(
            (nn.Linear(self.hidden_dim//2, 1))
        )


class CnnMlpNetwork4(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        share_out_channels: int = 32,
        mask_out_channels: int = 8,
        actor_out_channels: int = 8,
        critic_out_channels: int = 4,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
        )
        
        self.share_out_channels = share_out_channels
        self.mask_out_channels = mask_out_channels
        self.actor_out_channels = actor_out_channels
        self.critic_out_channels = critic_out_channels

        self.share_extractor = nn.Sequential(
            (nn.Conv2d(self.share_input_channels, 16, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(16, 16, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(16, 32, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(32, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
        )

        self.mask_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, mask_out_channels)
        self.actor_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, actor_out_channels)
        self.critic_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, critic_out_channels)

        self.mask_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, mask_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.mask_n_flatten, self.hidden_dim)),
            nn.ReLU(),
        )

        self.mask_net = nn.Sequential(
            (nn.Linear(self.hidden_dim, self.action_dim)),
            nn.Sigmoid(),
        )

        self.actor_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, actor_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.actor_n_flatten, self.hidden_dim)),
            nn.ReLU(),
        )

        self.actor_net = nn.Sequential(
            (nn.Linear(self.hidden_dim, self.action_dim)),
        )

        self.critic_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, critic_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.critic_n_flatten, self.hidden_dim//2)),
            nn.ReLU(),
        )

        self.critic_net = nn.Sequential(
            (nn.Linear(self.hidden_dim//2, 1))
        )
        