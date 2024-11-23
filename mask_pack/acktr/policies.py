# This file is here just to define MlpPolicy/CnnPolicy
# that work for ACKTR

from mask_pack.common.policies import CustomActorCriticPolicy

CnnMlpPolicy = CustomActorCriticPolicy
