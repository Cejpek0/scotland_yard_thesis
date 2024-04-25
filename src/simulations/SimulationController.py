import csv
import glob
import os
import random

import pandas
import ray

from src.game import scotland_yard_game_logic


class SimulationController:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.game = scotland_yard_game_logic.ScotlandYardGameLogic(training=False)
        self.current_game_id = 0

    def run(self, config):
        # number of simulations to run
        # turns_[Mr_X agent]_vs_[cop agent]
        turns_ppo_vs_ppo = config["turns_ppo_vs_ppo"]
        turns_random_vs_ppo = config["turns_random_vs_ppo"]
        turns_ppo_vs_random = config["turns_ppo_vs_random"]
        turns_random_vs_random = config["turns_random_vs_random"]

        # PPO_cop vs PPO_x 
        for i in range(turns_ppo_vs_ppo):
            self.simulate_game(scotland_yard_game_logic.DefinedAlgorithms.PPO,
                               scotland_yard_game_logic.DefinedAlgorithms.PPO)
            self.current_game_id += 1
        print("PPO vs PPO done")

        # Random_cop vs PPO_x
        for i in range(turns_random_vs_ppo):
            self.simulate_game(scotland_yard_game_logic.DefinedAlgorithms.RANDOM,
                               scotland_yard_game_logic.DefinedAlgorithms.PPO)
            self.current_game_id += 1
        print("Random vs PPO done")

        # PPO_cop vs Random_x
        for i in range(turns_ppo_vs_random):
            self.simulate_game(scotland_yard_game_logic.DefinedAlgorithms.PPO,
                               scotland_yard_game_logic.DefinedAlgorithms.RANDOM)
            self.current_game_id += 1
        print("PPO vs Random done")

        # Random_cop vs Random_x
        for i in range(turns_random_vs_random):
            self.simulate_game(scotland_yard_game_logic.DefinedAlgorithms.RANDOM,
                               scotland_yard_game_logic.DefinedAlgorithms.RANDOM)
            self.current_game_id += 1
        print("Random vs Random done")

        files = glob.glob(os.path.join(self.save_dir, "simulation*"))
        simulation_count = len(files)
        self.merge_csv_results(simulation_count)

    def run_train_simulation(self, config):
        ray.init()
        self.cop_model_ppo = None
        self.mrx_model_ppo = None

        self.policy_mrx_ppo = None
        self.policy_cop_ppo = None

        self.cop_model_dqn = None
        self.mrx_model_dqn = None

        def env_creator(env_config):
            return FakeEnv({})

        register_env("scotland_env", env_creator)

        def policy_mapping_fn(agent_id, episode, worker):
            return "mr_x_policy" if agent_id == "mr_x" else "cop_policy"

        ppo_config = PPOConfig()

        ppo_config["policies"] = {
            "mr_x_policy": MR_X_POLICY_SPEC,
            "cop_policy": COP_POLICY_SPEC,
        }

        ppo_config["policy_mapping_fn"] = policy_mapping_fn
        ppo_config.framework("torch")

        # Set the config object's env.
        algo_ppo = PPO(env="scotland_env", config=ppo_config)
        # check if trained policies exist
        directory = "trained_policies"
        assert os.path.exists(directory), f"Directory {directory} does not exist"
        algo_ppo.restore(directory)

        self.algo_ppo = algo_ppo

        dqn_config = (DQNConfig()
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
        }

        dqn_config["replay_config"] = replay_config
        dqn_config["exploration_config"] = exploration_config

        dqn_config.evaluation_config = {
            "evaluation_interval": 10,
            "evaluation_num_episodes": 10,
        }

        dqn_config["policies"] = {
            "mr_x_policy": MR_X_POLICY_SPEC,
            "cop_policy": COP_POLICY_SPEC,
        }

        dqn_config["policy_mapping_fn"] = policy_mapping_fn
        dqn_config.framework("torch")
        dqn_config["reuse_actors"] = True

        # Set the config object's env.
        algo_dqn = DQN(env="scotland_env", config=dqn_config)

        # check if trained policies exist
        directory = "trained_policies_dqn"

        assert os.path.exists(directory), f"Directory {directory} does not exist"
        algo_dqn.restore(directory)
        self.algo_dqn = algo_dqn

    def merge_csv_results(self, sim_number):
        game_files = glob.glob(os.path.join(self.save_dir, 'game*.csv'))
        merged_data = pandas.DataFrame()

        for file in game_files:
            data = pandas.read_csv(file)
            merged_data = merged_data.append(data, ignore_index=True)

        # Save the merged DataFrame to a new CSV file
        merged_data.to_csv(f"simulation_{sim_number}", index=False)
        pass

    def simulate_game(self, cop_algo, mr_x_algo):
        self.game.set_mrx_algo(mr_x_algo).set_cop_algo(cop_algo).reset()
        rounds_stats = {}
        while self.game.get_game_status() is scotland_yard_game_logic.GameStatus.ONGOING:
            self.game.play_turn()
            if self.game.playing_player_index == len(self.game.players) - 1:
                rounds_stats[self.game.get_current_round_number()] = self.get_round_statistics()

        game_stats = self.get_game_statistics()
        game_stats["game_id"] = self.current_game_id
        game_stats["mr_x_algo"] = mr_x_algo.value
        game_stats["cop_algo"] = cop_algo.value
        self.save_statistics(game_stats, rounds_stats)

    def save_statistics(self, game_stats, rounds_stats):
        file_name = self.save_dir + "/game_" + str(self.current_game_id) + ".csv"
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["game_stats", str(game_stats)])
            writer.writerow(["rounds_stats", str(rounds_stats)])

    def get_round_statistics(self):
        mr_x_avg_distance_to_cop = 0
        avg_distance_between_cops = 0

        for cop in self.game.get_cops():
            mr_x_avg_distance_to_cop += self.game.get_mr_x().get_distance_to(cop.get_position())
            cop_distance_to_other_cop = 0
            for other_cop in self.game.get_cops():
                cop_distance_to_other_cop = cop.get_distance_to(other_cop.get_position())
            avg_distance_between_cops += cop_distance_to_other_cop / 2

        mr_x_avg_distance_to_cop /= len(self.game.get_cops())
        return {"mr_x_avg_distance_to_cop": mr_x_avg_distance_to_cop,
                "avg_distance_between_cops": avg_distance_between_cops}

    def get_game_statistics(self):
        game_result = self.game.get_game_status().value
        return {"game_result": game_result}
