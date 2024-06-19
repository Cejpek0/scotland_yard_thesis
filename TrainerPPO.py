"""
File description: This file contains the TrainerPPO class, which is used to train and contain the PPO agents.

Author: Michal Cejpek (xcejpe05@stud.fit.vutbr.cz)
"""
import os

from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.util.client import ray
from ray.tune.registry import register_env
from src.game import scotland_yard_game_logic as scotland_yard_game
from src.environments.rlib.scotland_yard_environment import ScotlandYardEnvironment, policy_mapping_fn

from src.helper import verbose_print


class TrainerPPO:
    def __init__(self, directory="trained_models_ppo", verbose=False, simulation=False, playing=False):
        """
        Initialize the TrainerPPO object.
        :param directory: Directory to save the trained models.
        :param verbose: If True, print verbose output.
        :param simulation: Set to true if trainer is initialized for simulation purposes.
        Used to set the number of cpus per worker. And avoid multiple ray.init() calls.
        :param playing: Set to true if trainer is initialized for playing purposes.
        """
        if playing:
            assert os.path.exists(directory), "No trained policies found"
        self.directory = directory
        self.simulation = simulation
        self.verbose = verbose
        if not simulation and not playing:
            ray.init(num_gpus=0)
            # ray.init(num_gpus=1) for gpu support

        def env_creator(env_config):
            return ScotlandYardEnvironment({})

        register_env("scotland_env", env_creator)

        my_config = (PPOConfig()
                     .training())

        my_config["policies"] = {
            "mr_x_policy": scotland_yard_game.MR_X_POLICY_SPEC,
            "cop_policy": scotland_yard_game.COP_POLICY_SPEC,
        }

        my_config["policy_mapping_fn"] = policy_mapping_fn
        my_config["reuse_actors"] = True
        if not playing:
            my_config["num_rollout_workers"] = 4
        if simulation:
            my_config = my_config.resources(num_cpus_per_worker=1.2)
            # my_config = my_config.resources(num_gpu_per_worker=0.5) for gpu support
        my_config = my_config.framework("torch")
        self.config = my_config
        # Set the config object's env.
        algo = PPO(env="scotland_env", config=my_config)

        # check if trained policies exist
        if os.path.exists(directory):
            verbose_print("Loading policies", self.verbose)
            algo.restore(directory)
        self.algo = algo

    def train(self, number_of_iterations=1, save_interval=10):
        for i in range(number_of_iterations):
            verbose_print(f"Training iteration {i + 1} of {number_of_iterations}", self.verbose)
            train_results = self.algo.train()
            verbose_print(f"Information about training iteration {i + 1} of {number_of_iterations} done", self.verbose)
            verbose_print(f"Total time trained:{train_results['time_total_s']}", self.verbose)
            verbose_print(f"Number of episodes: {train_results['episodes_this_iter']} with average reward:{train_results['episode_reward_mean']}", self.verbose)
            if not self.simulation and i % save_interval == 0 and i != 0:
                verbose_print("Saving policies", self.verbose)
                self.save_export()
        verbose_print("Done", self.verbose)
        self.save_export()
        return self

    def save_export(self):
        self.algo.save(self.directory)
        # self.algo.get_policy("mr_x_policy").export_model(f"{self.directory}/trained_models/policy_model_mrx_ppo")
        # self.algo.get_policy("cop_policy").export_model(f"{self.directory}/trained_models/policy_model_cop_ppo")
        return self

    def cleanup(self):
        ray.shutdown()
        return self


if __name__ == "__main__":
    (TrainerPPO(directory="trained_models_ppo", verbose=True)
     .train(number_of_iterations=10, save_interval=5)
     .cleanup())
