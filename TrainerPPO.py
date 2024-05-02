import os

from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.util.client import ray
from ray.tune.registry import register_env
from src.game import scotland_yard_game_logic as scotland_yard_game
from src.environments.rlib.scotland_yard_environment import ScotlandYardEnvironment, policy_mapping_fn

from src.helper import verbose_print


class TrainerPPO:
    def __init__(self, directory="trained_policies_ppo", verbose=False, simulation=False, playing=False):
        self.directory = directory
        self.verbose = verbose
        if not simulation and not playing:
            ray.init(num_gpus=0)

        def env_creator(env_config):
            return ScotlandYardEnvironment({}, scotland_yard_game.DefinedAlgorithms.PPO, simulation=simulation)

        register_env("scotland_env", env_creator)

        my_config = (PPOConfig()
                     .training())

        my_config["policies"] = {
            "mr_x_policy": scotland_yard_game.MR_X_POLICY_SPEC,
            "cop_policy": scotland_yard_game.COP_POLICY_SPEC,
        }

        my_config["num_iterations"] = 20
        my_config["policy_mapping_fn"] = policy_mapping_fn
        my_config["reuse_actors"] = False
        if not playing:
            my_config["num_rollout_workers"] = 4
        if simulation:
            my_config = my_config.resources(num_cpus_per_worker=1.2)
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
            verbose_print(f"Episode reward mean:{self.algo.train()['episode_reward_mean']}", self.verbose)
            if i % save_interval == 0 and i != 0:
                verbose_print("Saving policies", self.verbose)
                self.save_export()
        verbose_print("Done", self.verbose)
        self.save_export()
        return self

    def save_export(self):
        self.algo.save(self.directory)
        self.algo.get_policy("mr_x_policy").export_model(f"{self.directory}/trained_models/policy_model_mrx_ppo")
        self.algo.get_policy("cop_policy").export_model(f"{self.directory}/trained_models/policy_model_cop_ppo")
        return self

    def cleanup(self):
        ray.shutdown()
        return self


if __name__ == "__main__":
    (TrainerPPO(directory="trained_policies_ppo", verbose=True)
     .train(number_of_iterations=10, save_interval=5)
     .cleanup())
