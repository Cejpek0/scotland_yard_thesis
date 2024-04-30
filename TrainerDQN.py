import os

from ray.rllib.algorithms import DQN, DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.util.client import ray
from ray.tune.registry import register_env
from src.game import scotland_yard_game_logic as scotland_yard_game
from src.environments.rlib.scotland_yard_environment import ScotlandYardEnvironment

from src.helper import verbose_print


class TrainerDQN:
    def __init__(self, max_iterations, directory="trained_policies_dqn", verbose=False, simulation=False, playing=False):
        self.directory = directory
        self.verbose = verbose
        if not simulation and not playing:
            ray.init()

        def policy_mapping_fn(agent_id, episode, worker):
            return "mr_x_policy" if agent_id == "mr_x" else "cop_policy"

        def env_creator(env_config):
            return ScotlandYardEnvironment({}, scotland_yard_game.DefinedAlgorithms.PPO, simulation=simulation)

        register_env("scotland_env", env_creator)

        my_config = (DQNConfig()
                     .training(model={"fcnet_hiddens": [64, 64]},
                               lr=0.0005,
                               gamma=0.999,
                               target_network_update_freq=10,
                               double_q=True,
                               num_atoms=1,
                               noisy=True,
                               n_step=3, )
                     .rollouts(batch_mode="complete_episodes"))

        if not playing and not simulation:
            my_config.exploration(explore=True, exploration_config={"type": "EpsilonGreedy", "initial_epsilon": 1.0,
                                                                "final_epsilon": 0.05,
                                                                "epsilon_timesteps": max_iterations * 1000})

        my_config.evaluation_config = {
            "evaluation_interval": 10,
            "evaluation_num_episodes": 10,
        }

        my_config["policies"] = {
            "mr_x_policy": scotland_yard_game.MR_X_POLICY_SPEC,
            "cop_policy": scotland_yard_game.COP_POLICY_SPEC,
        }

        my_config["policy_mapping_fn"] = policy_mapping_fn

        my_config["reuse_actors"] = False
        my_config.framework("torch")
        if simulation:
            my_config.resources(num_cpus_per_worker=0.2)

        # Set the config object's env.
        algo = DQN(env="scotland_env", config=my_config)

        # check if trained policies exist
        if os.path.exists(directory):
            verbose_print("Loading policies", self.verbose)
            algo.restore(directory)
        self.algo = algo
        self.config = my_config

    def adjust_rollout_fragment_length(self, iteration, start_length, max_length, total_iterations):
        progress = iteration / total_iterations
        return int(start_length + progress * (max_length - start_length))

    def train(self, number_of_iterations=1, save_interval=10):
        for i in range(number_of_iterations):
            verbose_print(f"Training iteration {i + 1} of {number_of_iterations}", self.verbose)
            verbose_print(f"Episode reward mean:{self.algo.train()['episode_reward_mean']}", self.verbose)
            verbose_print(
                f"Current epsilon: {self.algo.get_policy('mr_x_policy').get_exploration_state()['cur_epsilon']}",
                self.verbose)
            if i % save_interval == 0 and i != 0:
                verbose_print("Saving policies", self.verbose)
                self.save_export()
        verbose_print("Done", self.verbose)
        self.save_export()
        return self

    def save_export(self):
        self.algo.save(self.directory)
        return self

    def cleanup(self):
        ray.shutdown()
        return self


if __name__ == "__main__":
    TrainerDQN(max_iterations=50, directory="trained_policies_dqn", verbose=True).train(number_of_iterations=50, save_interval=50).cleanup()
    print("Done")
