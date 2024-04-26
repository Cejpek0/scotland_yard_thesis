import os

from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.util.client import ray
from ray.tune.registry import register_env
from src.game import scotland_yard_game_logic as scotland_yard_game
from src.environments.rlib.scotland_yard_environment import ScotlandYardEnvironment

from src.helper import verbose_print


class TrainerPPO:
    def __init__(self, directory="trained_policies_ppo", verbose=False, simulation=False):
        self.directory = directory
        self.verbose = verbose
        if not simulation:
            ray.init(num_gpus=1)

        def policy_mapping_fn(agent_id, episode, worker):
            return "mr_x_policy" if agent_id == "mr_x" else "cop_policy"

        def env_creator(env_config):
            return ScotlandYardEnvironment({}, scotland_yard_game.DefinedAlgorithms.PPO, simulation=simulation)

        register_env("scotland_env", env_creator)

        my_config = (PPOConfig()
                     .training())

        my_config["policies"] = {
            "mr_x_policy": scotland_yard_game.MR_X_POLICY_SPEC,
            "cop_policy": scotland_yard_game.COP_POLICY_SPEC,
        }

        my_config["policy_mapping_fn"] = policy_mapping_fn
        repeat = 20
        my_config["num_iterations"] = repeat
        my_config["num_rollout_workers"] = 4
        my_config["reuse_actors"] = True
        if simulation:
            my_config.resources(num_gpus=1, num_gpus_per_worker=0.2, num_cpus_per_worker=0.6)
        else:
            my_config.resources(num_gpus=1, num_gpus_per_worker=0.2)
        my_config.framework("torch")

        # Set the config object's env.
        algo = PPO(env="scotland_env", config=my_config)

        # check if trained policies exist
        if os.path.exists(directory):
            verbose_print("Loading policies", self.verbose)
            algo.restore(directory)
        self.algo = algo
        self.config = my_config

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
     .train(number_of_iterations=20, save_interval=1)
     .cleanup())
