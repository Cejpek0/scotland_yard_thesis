import os

from ray import train, tune
from ray.rllib.algorithms.dqn.dqn import DQNConfig, DQN
from ray.rllib.models import ModelCatalog
from ray.tune.stopper import MaximumIterationStopper
from ray.util.client import ray
from ray.tune.registry import register_env
from src.game import scotland_yard_game_logic as scotland_yard_game
from src.environments.rlib.scotland_yard_environment import ScotlandYardEnvironment
from ray.rllib.examples.models.centralized_critic_models import (
    TorchCentralizedCriticModel,
)

from tune_ppo import get_latest_checkpoint

if __name__ == "__main__":
    ray.init()


    def env_creator(env_config):
        return ScotlandYardEnvironment({}, scotland_yard_game.DefinedAlgorithms.DQN)


    register_env("scotland_env", env_creator)

    ModelCatalog.register_custom_model(
        "cc_model",
        TorchCentralizedCriticModel
    )

    stopper = MaximumIterationStopper(max_iter=100)
    directory = "C:/Users/cmich/Documents/Projects/school/scotland_yard_thesis/tuned_result_dqn"

    tune_config = {
        "env": "scotland_env",
        "framework": "torch",

        # Model configuration
        "model": {
            "custom_model": "cc_model",
            "fcnet_hiddens": tune.grid_search([
                [64, 32, 32],
                [64, 64, 32],
                [32, 64, 32],
            ]),
        },

        # Core DQN parameters
        "checkpoint_at_end": True,
        "checkpoint_freq": 10,
        "lr": tune.loguniform(1e-5, 1e-2),
        "gamma": 0.99,  # Keep this fixed in most cases
        "target_network_update_freq": 10,
        "double_q": True,
        "dueling": True,
        "num_atoms": 1,
        "noisy": True,
        "n_step": 3,

        # Replay buffer configuration
        "replay_buffer_config": {
            "_enable_replay_buffer_api": True,
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": tune.grid_search([10000, 50000, 100000]),
            "prioritized_replay_alpha": tune.uniform(0.3, 0.7),
            # Consider tuning prioritized_replay_beta 
            "prioritized_replay_eps": 1e-6
        },

        # Exploration configuration
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 100
        },

        "local_dir": directory,
        "train_batch_size": tune.choice([32, 64, 128]),
        "evaluation_interval": 10,
        "evaluation_num_episodes": 10
    }

    tune_config["policies"] = {
        "mr_x_policy": scotland_yard_game.MR_X_POLICY_SPEC,
        "cop_policy": scotland_yard_game.COP_POLICY_SPEC,
    }


    def policy_mapping_fn(agent_id, episode, worker):
        return "mr_x_policy" if agent_id == "mr_x" else "cop_policy"


    tune_config["policy_mapping_fn"] = policy_mapping_fn

    tune_config["reuse_actors"] = False

    checkpoint_path = get_latest_checkpoint()
    print(checkpoint_path)

#64,64,32,5000

    tuner = tune.Tuner(DQN,
                       tune_config=tune.TuneConfig(
                           metric="episode_reward_mean",
                           mode="max",
                           num_samples=1,
                       ),
                       run_config=train.RunConfig(stop={"training_iteration":    100, "time_total_s": 300}),
                       param_space=tune_config,
                       )

    analysis = tuner.fit()

    analysis.get_dataframe().to_csv("./tune_analysis.csv")

    ray.shutdown()
    print("Done")
    print(analysis)
