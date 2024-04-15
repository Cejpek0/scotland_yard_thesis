import os

from ray.rllib.algorithms.dqn.dqn import DQNConfig, DQN
from ray.rllib.models import ModelCatalog, ModelV2
from ray.tune.experiment.trial import ExportFormat
from ray.util.client import ray
from ray.tune.registry import register_env
from src.game import scotland_yard_game_logic as scotland_yard_game
from src.environments.rlib.scotland_yard_environment import ScotlandYardEnvironment
from ray.rllib.examples.models.centralized_critic_models import (
    TorchCentralizedCriticModel,
)

if __name__ == "__main__":
    ray.init(num_gpus=0)


    def env_creator(env_config):
        return ScotlandYardEnvironment({}, scotland_yard_game.DefinedAlgorithms.DQN)


    register_env("scotland_env", env_creator)

    ModelCatalog.register_custom_model(
        "cc_model",
        TorchCentralizedCriticModel
    )

    my_config = (DQNConfig()
                 .training(
                           lr=0.001,
                           gamma=0.99,
                           target_network_update_freq=10,
                           double_q=True,
                           dueling=True,
                           num_atoms=1,
                           noisy=True,
                           n_step=3, )
                 .rollouts(observation_filter="MeanStdFilter"))

    repeat = 1

    replay_config = {
        "_enable_replay_buffer_api": True,
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 50000,
        "prioritized_replay_alpha": 0.5,
        "prioritized_replay_beta": 0.5,
        "prioritized_replay_eps": 3e-6,
    }

    exploration_config = {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05,
        "epsilon_timesteps": repeat
    }

    my_config["replay_config"] = replay_config
    my_config["exploration_config"] = exploration_config

    my_config.evaluation_config = {
        "evaluation_interval": 10,
        "evaluation_num_episodes": 10,
    }

    my_config["policies"] = {
        "mr_x_policy": scotland_yard_game.MR_X_POLICY_SPEC,
        "cop_policy": scotland_yard_game.COP_POLICY_SPEC,
    }


    def adjust_rollout_fragment_length(iteration, start_length, max_length, total_iterations):
        progress = iteration / total_iterations
        return int(start_length + progress * (max_length - start_length))


    def policy_mapping_fn(agent_id, episode, worker):
        return "mr_x_policy" if agent_id == "mr_x" else "cop_policy"


    my_config["policy_mapping_fn"] = policy_mapping_fn

    my_config["num_rollout_workers"] = 4
    my_config["reuse_actors"] = True
    my_config.resources(num_gpus=0, num_gpus_per_worker=0)
    my_config["rollout_fragment_length"] = 100
    my_config.framework("tf")
    my_config.export_native_model_files = True

    # Set the config object's env.
    algo = DQN(env="scotland_env", config=my_config)

    # check if trained policies exist
    directory = "trained_policies_dqn"

    if os.path.exists(directory):
        print("Loading policies")
        algo.restore(directory)
    for i in range(repeat):
        print("Training iteration {} of {}".format(i + 1, repeat))
        current_length = adjust_rollout_fragment_length(i, 200, 2000, repeat)
        my_config["rollout_fragment_length"] = current_length
        print(f"Rollout fragment length: {current_length}")
        algo.config = my_config
        algo.train()
        if i % 4 == 0:
            print("Saving policies")
            algo.save(directory)
    algo.save(directory)

    algo.export_policy_model("trained_models/policy_model_mrx_dqn", "mr_x_policy")
    algo.export_policy_model("trained_models/policy_model_cop_dqn", "cop_policy")

    ray.shutdown()

    print("Done")
