from typing import Dict, List, Optional, Tuple, Type, Union, Any

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn
import math

class CustomMaxPool(nn.Module):
    def __init__(self, dim=1):
        super(CustomMaxPool, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.max(dim=self.dim)[0]
    
class CustomMeanPool(nn.Module):
    def __init__(self, dim=1):
        super(CustomMeanPool, self).__init__()
        self.dim = dim

    def forward(self, x):
        return th.mean(x, dim=self.dim)   

class ImplicitPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1600):
        super(ImplicitPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.encoding = th.zeros(max_len, embed_dim)                    # torch.Size([max_len, embed_dim])
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)   # torch.Size([max_len, 1])
        div_term = th.exp(th.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))    # torch.Size([embed_dim//2])
        self.encoding[:, 0::2] = th.sin(position * div_term)            # torch.Size([max_len, embed_dim//2])
        self.encoding[:, 1::2] = th.cos(position * div_term)            # torch.Size([max_len, embed_dim//2])
        self.encoding = self.encoding.unsqueeze(0)                      # torch.Size([1, max_len, embed_dim])

    def forward(self, x):
        seq_len, embed_dim = x.shape[1], x.shape[2]
        assert embed_dim == self.embed_dim, "input feature's embed_dim must be equal to the module's embed_dim"
        assert seq_len <= self.max_len, "seq_len must be less than max_len"
        return x + self.encoding[:seq_len, :].to(x.device)              # torch.Size([N, seq_len, embed_dim])

class BaseNetwork(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int = 100,
        normalize_images: bool = False,
        position_encode: bool = False,
        cnn_shortcut: bool = False,
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
        self.position_encode = position_encode
        self.cnn_shortcut = cnn_shortcut
        self.share_input_channels = observation_space.shape[0]

        # to be defined in the subclasses 
        self.share_extractor: nn.Sequential = None
        self.attention: nn.MultiheadAttention = None
        self.positional_encoding: ImplicitPositionalEncoding = None
        self.mask_net: nn.Sequential = None
        self.actor_net: nn.Sequential = None
        self.critic_net: nn.Sequential = None

        self.mask_n_flatten: int = None
        self.mask_extractor: nn.Sequential = None
        self.actor_n_flatten: int = None
        self.actor_extractor: nn.Sequential = None
        self.critic_n_flatten: int = None
        self.critic_extractor: nn.Sequential = None

    def forward(self, observations: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        cnn_f = self.share_extractor(observations)  # [N, share_out_channels, cW, cH]
        cnn_f = cnn_f.flatten(2).transpose(1, 2)

        if self.position_encode is True:
            cnn_f = self.positional_encoding(cnn_f)    # [N, cW*cH, share_out_channels]

        attn_output, _ = self.attention(cnn_f, cnn_f, cnn_f)
        if self.cnn_shortcut is True:
            mask_probs = self.mask_net(attn_output + cnn_f)               # torch.Size([N, action_dim])
            action_logits = self.actor_net(attn_output + cnn_f)           # torch.Size([N, action_dim])
            values = self.critic_net(attn_output + cnn_f)                 # torch.Size([N, 1])
        else:
            mask_probs = self.mask_net(attn_output)               # torch.Size([N, action_dim])
            action_logits = self.actor_net(attn_output)           # torch.Size([N, action_dim])
            values = self.critic_net(attn_output)                 # torch.Size([N, 1])
        return mask_probs, action_logits, values

    def forward_mask_probs(self, observations: th.Tensor) -> th.Tensor:
        cnn_f = self.share_extractor(observations)
        cnn_f = cnn_f.flatten(2).transpose(1, 2)    # torch.Size([N, cW*cH, output_channels])
        attn_output, _ = self.attention(cnn_f, cnn_f, cnn_f)     
        if self.cnn_shortcut is True:
            mask_probs = self.mask_net(attn_output + cnn_f)
        else:
            mask_probs = self.mask_net(attn_output)    
        return mask_probs

    def forward_action_logits(self, observations: th.Tensor) -> th.Tensor:
        cnn_f = self.share_extractor(observations)
        cnn_f = cnn_f.flatten(2).transpose(1, 2)    # torch.Size([N, cW*cH, output_channels])
        attn_output, _ = self.attention(cnn_f, cnn_f, cnn_f)
        if self.cnn_shortcut is True:
            action_logits = self.actor_net(attn_output + cnn_f)
        else:
            action_logits = self.actor_net(attn_output) 
        return action_logits
    
    def forward_critic(self, observations: th.Tensor) -> th.Tensor:
        cnn_f = self.share_extractor(observations)
        cnn_f = cnn_f.flatten(2).transpose(1, 2)    # torch.Size([N, cW*cH, output_channels])
        attn_output, _ = self.attention(cnn_f, cnn_f, cnn_f) 
        if self.cnn_shortcut is True:
            values = self.critic_net(attn_output + cnn_f)
        else:    
            values = self.critic_net(attn_output)      
        return values
    
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


class CnnAttenMlpNetwork1_v1(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        cnn_shortcut: bool = True,
        share_out_channels: int = 64,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
            cnn_shortcut=cnn_shortcut,
        )

        self.share_out_channels = share_out_channels
        self.attention = nn.MultiheadAttention(embed_dim=share_out_channels, **attention_kwargs)

        self.share_extractor = nn.Sequential(
            (nn.Conv2d(self.share_input_channels, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
        )

        self.mask_net = nn.Sequential(
            CustomMaxPool(dim=1), 
            nn.Linear(self.share_out_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Sigmoid(),
        )

        self.actor_net = nn.Sequential(
            CustomMaxPool(dim=1), 
            nn.Linear(self.share_out_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

        self.critic_net = nn.Sequential(
            CustomMaxPool(dim=1),
            nn.Linear(self.share_out_channels, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, 1),
        )

class CnnAttenMlpNetwork1_v2(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        cnn_shortcut: bool = True,
        share_out_channels: int = 64,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
            cnn_shortcut=cnn_shortcut,
        )

        self.share_out_channels = share_out_channels
        self.attention = nn.MultiheadAttention(embed_dim=share_out_channels, **attention_kwargs)

        self.share_extractor = nn.Sequential(
            (nn.Conv2d(self.share_input_channels, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
        )

        self.mask_net = nn.Sequential(
            CustomMeanPool(dim=1), 
            nn.Linear(self.share_out_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Sigmoid(),
        )

        self.actor_net = nn.Sequential(
            CustomMeanPool(dim=1), 
            nn.Linear(self.share_out_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

        self.critic_net = nn.Sequential(
            CustomMeanPool(dim=1),
            nn.Linear(self.share_out_channels, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, 1),
        )

class CnnAttenMlpNetwork1_v3(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        position_encode: bool = True,
        cnn_shortcut: bool = True,
        share_out_channels: int = 64,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
            position_encode=position_encode,
            cnn_shortcut=cnn_shortcut,
        )

        self.share_out_channels = share_out_channels
        self.attention = nn.MultiheadAttention(embed_dim=share_out_channels, **attention_kwargs)

        self.positional_encoding = ImplicitPositionalEncoding(embed_dim=share_out_channels, max_len=self.action_dim)

        self.share_extractor = nn.Sequential(
            (nn.Conv2d(self.share_input_channels, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
        )

        self.mask_net = nn.Sequential(
            CustomMeanPool(dim=1), 
            nn.Linear(self.share_out_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Sigmoid(),
        )

        self.actor_net = nn.Sequential(
            CustomMeanPool(dim=1), 
            nn.Linear(self.share_out_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

        self.critic_net = nn.Sequential(
            CustomMeanPool(dim=1),
            nn.Linear(self.share_out_channels, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, 1),
        )

class CnnMlpNetwork1(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        share_out_channels: int = 64,
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
            (nn.Conv2d(self.share_input_channels, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
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

    def forward(self, observations: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        share_features = self.share_extractor(observations)

        latent_mk = self.mask_extractor(share_features)
        mask_probs = self.mask_net(latent_mk)

        latent_pi = self.actor_extractor(share_features)
        action_logits = self.actor_net(latent_pi)

        latent_vf = self.critic_extractor(share_features)
        values = self.critic_net(latent_vf)
        return mask_probs, action_logits, values

    def forward_action_logits(self, observations: th.Tensor) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_pi = self.actor_extractor(share_features)
        return self.actor_net(latent_pi)
    
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
    
class CnnMlpNetwork2(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        share_out_channels: int = 64,
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
            (nn.Conv2d(self.share_input_channels, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
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

    def forward(self, observations: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        share_features = self.share_extractor(observations)

        latent_mk = self.mask_extractor(share_features)
        mask_probs = self.mask_net(latent_mk)

        latent_pi = self.actor_extractor(share_features)
        action_logits = self.actor_net(latent_pi)

        latent_vf = self.critic_extractor(share_features)
        values = self.critic_net(latent_vf)
        return mask_probs, action_logits, values

    def forward_action_logits(self, observations: th.Tensor) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_pi = self.actor_extractor(share_features)
        return self.actor_net(latent_pi)
    
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
    
class CnnMlpNetwork3(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        share_out_channels: int = 64,
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
            (nn.Conv2d(self.share_input_channels, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
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

    def forward(self, observations: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        share_features = self.share_extractor(observations)

        latent_mk = self.mask_extractor(share_features)
        mask_probs = self.mask_net(latent_mk)

        latent_pi = self.actor_extractor(share_features)
        action_logits = self.actor_net(latent_pi)

        latent_vf = self.critic_extractor(share_features)
        values = self.critic_net(latent_vf)
        return mask_probs, action_logits, values

    def forward_action_logits(self, observations: th.Tensor) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_pi = self.actor_extractor(share_features)
        return self.actor_net(latent_pi)
    
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
    
class CnnMlpNetwork4(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        share_out_channels: int = 64,
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
            (nn.Conv2d(self.share_input_channels, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
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

    def forward(self, observations: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        share_features = self.share_extractor(observations)

        latent_mk = self.mask_extractor(share_features)
        mask_probs = self.mask_net(latent_mk)

        latent_pi = self.actor_extractor(share_features)
        action_logits = self.actor_net(latent_pi)

        latent_vf = self.critic_extractor(share_features)
        values = self.critic_net(latent_vf)
        return mask_probs, action_logits, values

    def forward_action_logits(self, observations: th.Tensor) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_pi = self.actor_extractor(share_features)
        return self.actor_net(latent_pi)
    
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
