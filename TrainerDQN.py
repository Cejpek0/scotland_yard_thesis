import os

from ray.rllib.algorithms import DQN, DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.util.client import ray
from ray.tune.registry import register_env
from src.game import scotland_yard_game_logic as scotland_yard_game
from src.environments.rlib.scotland_yard_environment import ScotlandYardEnvironment

from src.helper import verbose_print


class TrainerDQN:
    def __init__(self, max_iterations, directory="trained_policies_dqn", verbose=False, simulation=False):
        self.directory = directory
        self.verbose = verbose
        if not simulation:
            ray.init(num_gpus=1)

        def policy_mapping_fn(agent_id, episode, worker):
            return "mr_x_policy" if agent_id == "mr_x" else "cop_policy"

        def env_creator(env_config):
            return ScotlandYardEnvironment({}, scotland_yard_game.DefinedAlgorithms.PPO, simulation=simulation)

        register_env("scotland_env", env_creator)

        my_config = (DQNConfig()
                     .training(
            lr=0.0005,
            gamma=0.999,
            target_network_update_freq=10,
            double_q=True,
            dueling=True,
            num_atoms=1,
            noisy=True,
            n_step=3, )
                     .rollouts(observation_filter="MeanStdFilter").exploration(explore=True))

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
            "epsilon_timesteps": max_iterations
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

        my_config["policy_mapping_fn"] = policy_mapping_fn

        my_config["num_rollout_workers"] = 4
        my_config["reuse_actors"] = True
        my_config["rollout_fragment_length"] = 100
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
            current_length = self.adjust_rollout_fragment_length(i, 200, 2000, number_of_iterations)
            self.config["rollout_fragment_length"] = current_length
            verbose_print(f"Episode reward mean:{self.algo.train()['episode_reward_mean']}", self.verbose)
            verbose_print(f"Current rollout fragment length: {current_length}", self.verbose)
            verbose_print(f"Current epsilon: {self.algo.get_policy('mr_x_policy').exploration.get_state()['cur_epsilon']}", self.verbose)
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
    (TrainerDQN(max_iterations=100, directory="trained_policies_dqn", verbose=True)
               .train(number_of_iterations=100, save_interval=10)
               .cleanup())
    print("Done")
