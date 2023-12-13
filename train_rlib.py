import argparse

from ray import air, tune
from gymnasium import spaces
from ray.rllib.examples.env.action_mask_env import ActionMaskEnv
from ray.rllib.examples.models.parametric_actions_model import TorchParametricActionsModel, ParametricActionsModel
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import check_env
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.util.client import ray
import numpy as np

import scotland_yard_game
from environments.rlib.scotland_yard_environment_1v1 import ScotlandYardEnvironment1v1
from pettingzoo.test import api_test

# from pettingzoo.sisl import waterworld_v4
# Based on code from github.com/parametersharingmadrl/parametersharingmadrl


if __name__ == "__main__":
    ray.init()
    from ray.tune.registry import register_env


    def env_creator(env_config):
        return ScotlandYardEnvironment1v1({})  # return an env instance


    register_env("scotland_env", env_creator)


    config = PPOConfig({"multiagent": {
        "policies": {
            # Use the PolicySpec namedtuple to specify an individual policy:
            "mr_x": PolicySpec(
                observation_space=spaces.Box(low=np.array([
                    0,  # current turn
                    0,  # max turns
                    0,  # next reveal
                    0,  # position x
                    0,  # position y
                    0,  # position x of cop
                    0,  # position y or cop
                    -1,  # last known position x
                    -1,  # last known position y
                    0  # distance to cop
                ]), high=np.array([
                    scotland_yard_game.MAX_NUMBER_OF_TURNS,  # current turn
                    scotland_yard_game.MAX_NUMBER_OF_TURNS,  # max turns
                    scotland_yard_game.MAX_NUMBER_OF_TURNS,  # next reveal
                    scotland_yard_game.GRID_SIZE,  # position x
                    scotland_yard_game.GRID_SIZE,  # position y
                    scotland_yard_game.GRID_SIZE,  # position x of cop
                    scotland_yard_game.GRID_SIZE,  # position y or cop
                    scotland_yard_game.GRID_SIZE,  # last known position x
                    scotland_yard_game.GRID_SIZE,  # last known position y
                    scotland_yard_game.GRID_SIZE * 2  # distance to cop
                ]), dtype=np.float32),
                action_space=spaces.Discrete(4),
                config={"gamma": 0.85},  # use main config plus <- this override here
            ),
            "cop_1": PolicySpec(
                observation_space=spaces.Box(low=np.array([
                    0,  # current turn
                    0,  # max turns
                    0,  # next reveal
                    0,  # position x
                    0,  # position y
                    -1,  # last known position x
                    -1,  # last known position y
                ]), high=np.array([
                    scotland_yard_game.MAX_NUMBER_OF_TURNS,  # current turn
                    scotland_yard_game.MAX_NUMBER_OF_TURNS,  # max turns
                    scotland_yard_game.MAX_NUMBER_OF_TURNS,  # next reveal
                    scotland_yard_game.GRID_SIZE,  # position x
                    scotland_yard_game.GRID_SIZE,  # position y
                    scotland_yard_game.GRID_SIZE,  # last known position x
                    scotland_yard_game.GRID_SIZE,  # last known position y
                ]), dtype=np.float32),
                action_space=spaces.Discrete(4),
                config={"gamma": 0.95},  # use main config plus <- this override here
            ),
        },
        "policy_mapping_fn":
            lambda agent_id, episode, worker, **kwargs:
            "mr_x"
            if agent_id.startswith("mr_x")
            else "cop_1"
    }})
    # Print out some default values.
    print(config.clip_param)
    # Update the config object.

    # Set the config object's env.
    config = config.environment(env="scotland_env")
    algo = PPO(env="scotland_env", config=config)

    while True:
        print(algo.train())
