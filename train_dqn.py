import os

from ray.rllib.algorithms.dqn.dqn import DQNConfig, DQN
from ray.rllib.models import ModelCatalog
from ray.util.client import ray
from ray.tune.registry import register_env
from src.game import scotland_yard_game_logic as scotland_yard_game
from src.environments.rlib.scotland_yard_environment import ScotlandYardEnvironment
from ray.rllib.examples.models.centralized_critic_models import (
    TorchCentralizedCriticModel,
)

if __name__ == "__main__":
    ray.init(num_gpus=1)


    def env_creator(env_config):
        return ScotlandYardEnvironment({})


    register_env("scotland_env", env_creator)

    ModelCatalog.register_custom_model(
        "cc_model",
        TorchCentralizedCriticModel
    )

    my_config = (DQNConfig()
                 .training(model={"custom_model": "cc_model"}, )
                 .rollouts(observation_filter="MeanStdFilter"))

    replay_config = {
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 60000,
        "prioritized_replay_alpha": 0.5,
        "prioritized_replay_beta": 0.5,
        "prioritized_replay_eps": 3e-6,
    }

    exploration_config = {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05
    }

    my_config["replay_config"] = replay_config
    my_config["exploration_config"] = exploration_config

    my_config["policies"] = {
        "mr_x_policy": scotland_yard_game.MR_X_POLICY_SPEC,
        "cop_policy": scotland_yard_game.COP_POLICY_SPEC,
    }


    def policy_mapping_fn(agent_id, episode, worker):
        return "mr_x_policy" if agent_id == "mr_x" else "cop_policy"


    my_config["policy_mapping_fn"] = policy_mapping_fn

    my_config["num_iterations"] = 100000
    my_config["num_rollout_workers"] = 4
    my_config["reuse_actors"] = False
    my_config.resources(num_gpus=1, num_gpus_per_worker=0.2)
    my_config["rollout_fragment_length"] = 2500
    my_config.framework("torch")

    # Set the config object's env.
    algo = DQN(env="scotland_env", config=my_config)

    repeat = 100
    # check if trained policies exist
    directory = "trained_policies_dqn"

    if os.path.exists(directory):
        print("Loading policies")
        algo.restore(directory)
    for i in range(repeat):
        print("Training iteration {} of {}".format(i + 1, repeat))
        algo.train()
        if i % 4 == 0:
            print("Saving policies")
            algo.save(directory)
    algo.save(directory)

    ray.shutdown()

    print("Done")
