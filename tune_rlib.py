import numpy as np
from gymnasium import spaces
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.policy import PolicySpec
from ray.util.client import ray
from ray.tune.registry import register_env
import scotland_yard_game
from environments.rlib.scotland_yard_environment_1v1 import ScotlandYardEnvironment1v1

if __name__ == "__main__":
    ray.init(num_gpus=1)

    @ray.remote(num_gpus=1)
    def use_gpu():
        import tensorflow as tf

        # Create a TensorFlow session. TensorFlow will restrict itself to use the
        # GPUs specified by the CUDA_VISIBLE_DEVICES environment variable.
        tf.Session()

    def env_creator(env_config):
        return ScotlandYardEnvironment1v1({})  # return an env instance


    register_env("scotland_env", env_creator)

    repeat = 10

    tune_config = {
        "env": "scotland_env",
        "num_workers": 0,
        "num_gpus": 1,
        "num_gpus_per_worker": 1,
        "num_envs_per_worker": 1,
        "model": {
            "fcnet_hiddens": [512, 512, 256],
            "fcnet_activation": "relu",
        },
        "lr": 3e-4,
        "optimization": {
            "optimizer": "adam",
            "adam_epsilon": 1e-8,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
        },
        "gamma": 0.99,
        "num_sgd_iter": 10,
        "sgd_minibatch_size": 500,
        "rollout_fragment_length": 500,
        "train_batch_size": 4000,
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4,
        "buffer_size": 500000,
        "stop": {"episodes_total": 100},
        "exploration_config": {},
        "multiagent": {
            "policies": {
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
            },
            "policy_mapping_fn": lambda agent_id, episode, worker, *kw: "mr_x_policy" if agent_id == "mr_x" else "cop_policy"
        }
    }

    result = tune.run(
        run_or_experiment=PPO,
        config=tune_config,
        stop={"training_iteration": repeat},
        checkpoint_at_end=True,
        checkpoint_freq=1,
        checkpoint_score_attr="episode_reward_mean",
        name="PPO",
    )
    print(result)
    ray.shutdown()
