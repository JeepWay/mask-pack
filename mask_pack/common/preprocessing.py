import warnings
from typing import Dict, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces
from mask_pack.common.constants import BIN, MASK

def preprocess_obs(
    obs: Union[th.Tensor, Dict[str, th.Tensor]],
    observation_space: spaces.Space,
    normalize_images: bool = False,
) -> Union[th.Tensor, Dict[str, th.Tensor]]:
    """
    Preprocess observation to be to a neural network.
    For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
    For discrete observations, it create a one hot vector.

    :param obs: Observation
    :param observation_space:
    :param normalize_images: Whether to normalize images or not
        (True by default)
    :return:
    """
    if isinstance(observation_space, spaces.Dict):
        # Do not modify by reference the original observation
        assert isinstance(obs, Dict), f"Expected dict, got {type(obs)}"
        preprocessed_obs = {}
        preprocessed_obs[BIN] = preprocess_obs(obs[BIN], observation_space[BIN], normalize_images=normalize_images)
        preprocessed_obs[MASK] = preprocess_obs(obs[MASK], observation_space[MASK], normalize_images=False)
        # for key, _obs in obs.items():
        #     preprocessed_obs[key] = preprocess_obs(_obs, observation_space[key], normalize_images=normalize_images)
        return preprocessed_obs  # type: ignore[return-value]

    assert isinstance(obs, th.Tensor), f"Expecting a torch Tensor, but got {type(obs)}"

    if isinstance(observation_space, spaces.Box):
        if normalize_images:
            _, bin_w, bin_h = observation_space.shape
            obs = obs.float().clone() 
            obs[:, 1, :, :] = obs[:, 1, :, :] / bin_w
            obs[:, 2, :, :] = obs[:, 2, :, :] / bin_h
            return obs
        return obs.float()
    else:
        raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")
