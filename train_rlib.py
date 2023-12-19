import os

import numpy as np
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.policy.policy import PolicySpec
from ray.util.client import ray
from ray.tune.registry import register_env
import scotland_yard_game
from environments.rlib.scotland_yard_environment_1v1 import ScotlandYardEnvironment1v1

if __name__ == "__main__":
    import GPUtil

    gpus = GPUtil.getGPUs()
    print("Num GPUs Available:", len(gpus))
    print(gpus[0].name)


    ray.init(num_gpus=1)

    
    def env_creator(env_config):
        return ScotlandYardEnvironment1v1({})  # return an env instance


    register_env("scotland_env", env_creator)

    my_config = PPOConfig()
    my_config["policies"] = {
        "mr_x_policy": PolicySpec(
            observation_space=spaces.Box(
                low=np.array([
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
                ]),
                high=np.array([
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
                ]),
                dtype=np.float32
            ),
            action_space=spaces.Discrete(4),
        ),
        "cop_policy": PolicySpec(
            observation_space=spaces.Box(
                low=np.array([
                    0,  # current turn
                    0,  # max turns
                    0,  # next reveal
                    0,  # position x
                    0,  # position y
                    -1,  # last known position x
                    -1,  # last known position y
                ]),
                high=np.array([
                    scotland_yard_game.MAX_NUMBER_OF_TURNS,  # current turn
                    scotland_yard_game.MAX_NUMBER_OF_TURNS,  # max turns
                    scotland_yard_game.MAX_NUMBER_OF_TURNS,  # next reveal
                    scotland_yard_game.GRID_SIZE,  # position x
                    scotland_yard_game.GRID_SIZE,  # position y
                    scotland_yard_game.GRID_SIZE,  # last known position x
                    scotland_yard_game.GRID_SIZE,  # last known position y
                ]),
                dtype=np.float32
            ),
            action_space=spaces.Discrete(4),
        ),
    }
    my_config["policy_mapping_fn"] = \
        lambda agent_id, episode, worker, *kw: "mr_x_policy" if agent_id == "mr_x" else "cop_policy"

    my_config["num_iterations"] = 10
    my_config["num_rollout_workers"] = 1
    my_config["reuse_actors"] = True
    my_config.resources(num_gpus=1, num_gpus_per_worker=0.5)
    my_config.framework("torch")


    # Set the config object's env.
    algo = PPO(env="scotland_env", config=my_config)

    repeat = 10
    # check if trained policies exist
    directory = "trained_policies"


    if os.path.exists(directory):
        algo.restore(directory)
    for i in range(repeat):
        print("Training iteration {} of {}".format(i + 1, repeat))
        print(algo.train())
        if i % 5 == 0:
            print("Saving policies")
            algo.save(directory)            
    algo.save(directory)

    ray.shutdown()
    
    print("Done")
