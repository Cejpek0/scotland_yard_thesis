import os

from ray.rllib.algorithms import DQN, DQNConfig
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.util.client import ray
from ray.tune.registry import register_env
from src.game import scotland_yard_game_logic as scotland_yard_game
from src.environments.rlib.scotland_yard_environment import ScotlandYardEnvironment

from src.helper import verbose_print


class TrainerDQN:
    def __init__(self, max_iterations, directory="trained_policies_dqn", verbose=False, simulation=False,
                 playing=False):
        self.directory = directory
        self.verbose = verbose
        if not simulation and not playing:
            ray.init(num_gpus=0)

        def policy_mapping_fn(agent_id, episode, worker):
            return "mr_x_policy" if agent_id == "mr_x" else "cop_policy"

        def env_creator(env_config):
            return ScotlandYardEnvironment({}, scotland_yard_game.DefinedAlgorithms.PPO, simulation=simulation)

        register_env("scotland_env", env_creator)

        my_config = (DQNConfig()
                     .training(model={"fcnet_hiddens": [128, 64]},
                               lr=0.0002,
                               gamma=0.99,
                               target_network_update_freq=200,
                               double_q=True,
                               num_atoms=1,
                               noisy=True,
                               n_step=5)
                     .rollouts(batch_mode="complete_episodes"))

        if not playing:
            my_config = my_config.exploration(explore=True,
                                              exploration_config={"type": "EpsilonGreedy", "initial_epsilon": 0.8,
                                                                  "final_epsilon": 0.05,
                                                                  "epsilon_timesteps": max_iterations / 10 * 9 * 1000})
        else:
            my_config = my_config.exploration(explore=True,
                                              exploration_config={"type": "EpsilonGreedy", "initial_epsilon": 0.05,
                                                                  "final_epsilon": 0.05,
                                                                  "epsilon_timesteps": 1})
        replay_config = {
            "_enable_replay_buffer_api": True,
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": 100000,
            "prioritized_replay_alpha": 0.5,
            "prioritized_replay_beta": 0.5,
            "prioritized_replay_eps": 3e-6,
        }

        my_config.replay_buffer_config = replay_config

        my_config.evaluation_config = {
            "evaluation_interval": 10,
            "evaluation_num_episodes": 10,
        }

        my_config["policies"] = {
            "mr_x_policy": scotland_yard_game.MR_X_POLICY_SPEC,
            "cop_policy": scotland_yard_game.COP_POLICY_SPEC,
        }

        my_config["policy_mapping_fn"] = policy_mapping_fn
        my_config["entropy_coeff"] = 0.01
        my_config["reuse_actors"] = False
        if not playing:
            my_config["num_rollout_workers"] = 4
        my_config["train_batch_size"] = 128
        my_config.framework("torch")
        if simulation:
            my_config = my_config.resources(num_cpus_per_worker=0.8)
        self.play_config = my_config.copy()
        # Set the config object's env.
        algo = DQN(env="scotland_env", config=my_config)

        # check if trained policies exist
        if os.path.exists(directory):
            verbose_print("Loading policies", self.verbose)
            algo.restore(directory)
        self.algo = algo
        self.config = my_config

        self.play_config = self.play_config.exploration(explore=True, exploration_config={"type": "EpsilonGreedy",
                                                                                          "initial_epsilon": 0.05,
                                                                                          "final_epsilon": 0.05,
                                                                                          "epsilon_timesteps": 1})

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
    TrainerDQN(max_iterations=200, directory="trained_policies_dqn", verbose=True).train(number_of_iterations=200,
                                                                                          save_interval=10).cleanup()
    print("Done")
