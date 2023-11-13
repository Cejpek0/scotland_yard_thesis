import numpy
import os

from ray.rllib.utils.pre_checks.env import check_multiagent_environments
from ray.tune import register_env

from scotland_yard_environment import ScotlandYardEnvironment
from ray.rllib.algorithms.ppo import PPOConfig

if __name__ == '__main__':
    env = ScotlandYardEnvironment({})
    group_env = env.with_agent_groups({"mr_x": ["mr_x"], "cops": ["cop_" + cop.number.__str__() for cop in env.game.get_cops()]})


    config = PPOConfig()
    config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)
    config = config.resources(num_gpus=0)
    config = config.rollouts(num_rollout_workers=4)
    print(config.to_dict())
    check_multiagent_environments(env)
    exit(0)
    # Build a Algorithm object from the config and run 1 training iteration.
    register_env('scotland', lambda config: env)
    algo = config.build("scotland")
    while True:
        print(algo.train())
