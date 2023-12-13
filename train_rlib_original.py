import argparse

from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from environments.rlib.scotland_yard_environment_1v1 import ScotlandYardEnvironment1v1
from pettingzoo.test import api_test
#from pettingzoo.sisl import waterworld_v4
# Based on code from github.com/parametersharingmadrl/parametersharingmadrl


if __name__ == "__main__":
    config = PPOConfig()
    # Print out some default values.
    print(config.clip_param)
    # Update the config object.
    config.training(
        lr=tune.grid_search([0.001, 0.0001]), clip_param=0.2
    )

    # Set the config object's env.
    env = ScotlandYardEnvironment1v1
    config = config.environment(env=env)
    # Use to_dict() to get the old-styltrain_rlib.pye python config dict
    # when running with tune.
    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(stop={"episode_reward_mean": 200}),
        param_space=config.to_dict(),
    ).fit()