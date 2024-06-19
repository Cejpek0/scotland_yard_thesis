"""
File description: This file contains the SimulationController class, which is responsible for running the simulations
of the game. It is responsible for running the simulations, saving the results, and merging the results into a single
file.

Author: Michal Cejpek (xcejpe05@stud.fit.vutbr.cz)
"""
import glob
import os
from datetime import datetime

import pandas
import ray

from TrainerDQN import TrainerDQN
from TrainerPPO import TrainerPPO
from src.game import scotland_yard_game_logic
from src.game.scotland_yard_game_logic import DefinedAlgorithms
from src.helper import verbose_print


class SimulationController:
    def __init__(self, save_dir, verbose=False, experiment_training_iteration_count=1000,
                 test_games_every_n_trainings=10, test_games_count_per_pause=50, simulation_experiment=False):
        """
        Description: This function initializes the SimulationController.
        :param save_dir: str - directory to save the results to
        :param verbose: bool - flag if simulation should print during the run
        :param experiment_training_iteration_count: int - number of training iterations
        :param test_games_every_n_trainings: int - start game simulations every n trainings
        :param test_games_count_per_pause: int - number of test games to be played per simulation step
        :param simulation_experiment: bool - flag if the simulation experiment should be run. If so, all other parameters are ignored.
        """
        
        ray.init(num_gpus=0, num_cpus=8)
        self.verbose = verbose
        self.simulation_experiment = simulation_experiment
        verbose_print("Simulation Controller initializing", self.verbose)
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # directories with all possible combinations of cop and mr_x agents
        self.experiment_directories = ["cop_dqn_mrx_ppo", "cop_ppo_mrx_dqn", "cop_dqn_mrx_dqn",
                                       "cop_ppo_mrx_ppo", "cop_random_mrx_ppo", "cop_ppo_mrx_random",
                                       "cop_dqn_mrx_random", "cop_random_mrx_dqn", "cop_random_mrx_random"]
        self.simulation_experiment_dir = self.save_dir + "/simulation_experiment/"
        
        # check if directories exist, if not create them
        if not os.path.exists(self.simulation_experiment_dir):
            os.makedirs(self.simulation_experiment_dir)
        for directory in self.experiment_directories:
            if not os.path.exists(self.save_dir + "/train_experiment/" + directory):
                os.makedirs(self.save_dir + "/train_experiment/" + directory)
            if not os.path.exists(self.simulation_experiment_dir + directory):
                os.makedirs(self.simulation_experiment_dir + directory)
        self.game = scotland_yard_game_logic.ScotlandYardGameLogic()
        self.current_game_id = 0
        self.train_experiment_training_count = experiment_training_iteration_count
        self.test_games_every_n_trainings = test_games_every_n_trainings
        self.test_games_every_n_trainings_mapping = {10: 1, 50: 10, 500: 25, 1000: 50}
        self.number_of_test_games_per_pause = test_games_count_per_pause
        if not simulation_experiment:
            # load the trainers for training simulation
            self.ppo_trainer = TrainerPPO(simulation=True)
            self.dqn_trainer = TrainerDQN(self.train_experiment_training_count, simulation=True)
        else:
            # load the trainers for simulation experiment
            self.ppo_trainer = TrainerPPO(playing=True)
            self.dqn_trainer = TrainerDQN(self.train_experiment_training_count, playing=True)
            self.dqn_trainer.algo.reset_config(self.dqn_trainer.play_config)
        verbose_print("Simulation Controller initialized", self.verbose)
        self.simulation_start_time = None
        self.last_simulation_time = None

    def get_current_test_game_every_n_trainings(self, current_training_iteration):
        """
        Description: This function returns the number of test games to be played every n trainings.
        :param current_training_iteration: int - current training iteration
        :return: int - number of test games to be played every n trainings
        """
        test_games_every_n_trainings = 50
        for key in self.test_games_every_n_trainings_mapping.keys():
            if current_training_iteration <= key:
                if test_games_every_n_trainings > self.test_games_every_n_trainings_mapping[key]:
                    test_games_every_n_trainings = self.test_games_every_n_trainings_mapping[key]
        return test_games_every_n_trainings

    def run(self, config):
        """
        Description: This function runs the simulation experiment based on the configuration.
        :param config: dict - configuration of the simulation
        :return: SimulationController - instance of the SimulationController
        """
        games_ppo_vs_ppo = config["games_ppo_vs_ppo"]
        games_random_vs_ppo = config["games_random_vs_ppo"]
        games_ppo_vs_random = config["games_ppo_vs_random"]
        games_random_vs_random = config["games_random_vs_random"]

        self.simulation_start_time = datetime.now()

        # PPO_cop vs PPO_x
        for i in range(games_ppo_vs_ppo):
            self.simulate_game(self.ppo_trainer.algo, DefinedAlgorithms.PPO,
                               self.ppo_trainer.algo, DefinedAlgorithms.PPO)
            self.current_game_id += 1
        print("PPO vs PPO done")

        # Random_cop vs PPO_x
        for i in range(games_random_vs_ppo):
            self.simulate_game(None, DefinedAlgorithms.RANDOM,
                               self.ppo_trainer.algo, DefinedAlgorithms.PPO)
            self.current_game_id += 1
        print("Random vs PPO done")

        # PPO_cop vs Random_x
        for i in range(games_ppo_vs_random):
            self.simulate_game(self.ppo_trainer.algo, DefinedAlgorithms.PPO,
                               None, DefinedAlgorithms.RANDOM)
            self.current_game_id += 1
        print("PPO vs Random done")

        # Random_cop vs Random_x
        for i in range(games_random_vs_random):
            self.simulate_game(None, DefinedAlgorithms.RANDOM,
                               None, DefinedAlgorithms.RANDOM)
            self.current_game_id += 1
        print("Random vs Random done")

        files = glob.glob(os.path.join(self.save_dir + "/", "simulation*"))
        simulation_count = len(files)
        self.last_simulation_time = datetime.now()
        self.merge_csv_results(simulation_count)
        verbose_print(f"Merge took {datetime.now() - self.last_simulation_time}", self.verbose)
        self.simulation_start_time = None

    def batch_simulation(self, number_of_simulations, cop_algo, cop_selected_algo: DefinedAlgorithms, mr_x_algo,
                         mr_x_selected_algo: DefinedAlgorithms, use_game_id=True, save_dir=None):
        """
        Description: This function runs a batch of simulations based on the configuration.
        :param number_of_simulations: int - number of simulations to be run
        :param cop_algo: Trainer - instance of the cop agent
        :param cop_selected_algo: DefinedAlgorithms - selected algorithm for the cop agent
        :param mr_x_algo: Trainer - instance of the mr_x agent
        :param mr_x_selected_algo: DefinedAlgorithms - selected algorithm for the mr_x agent
        :param use_game_id: bool - flag if the game id should be used
        :param save_dir: str - directory to save the results
        """
        for i in range(number_of_simulations):
            if not use_game_id:
                self.current_game_id = i + 1
            self.simulate_game(cop_algo,
                               cop_selected_algo,
                               mr_x_algo,
                               mr_x_selected_algo,
                               save_dir)
            if use_game_id:
                self.current_game_id = self.current_game_id + 1

    def simulate_all_variants(self, save_dir=None):
        """
        Description: This function runs simulations for all possible combinations of cop and mr_x agents.
        :param save_dir: str - directory to save the results
        :return: None
        """
        now = datetime.now()
        self.batch_simulation(self.number_of_test_games_per_pause,
                              self.dqn_trainer.algo, DefinedAlgorithms.DQN,
                              self.ppo_trainer.algo, DefinedAlgorithms.PPO,
                              use_game_id=False, save_dir=save_dir + "cop_dqn_mrx_ppo")
        verbose_print(f"Simulations took {datetime.now() - now}", verbose=self.verbose)
        now = datetime.now()
        self.batch_simulation(self.number_of_test_games_per_pause,
                              self.ppo_trainer.algo, DefinedAlgorithms.PPO,
                              self.dqn_trainer.algo, DefinedAlgorithms.DQN,
                              use_game_id=False, save_dir=save_dir + "cop_ppo_mrx_dqn")
        verbose_print(f"Simulations took {datetime.now() - now}", verbose=self.verbose)
        now = datetime.now()
        self.batch_simulation(self.number_of_test_games_per_pause,
                              self.dqn_trainer.algo, DefinedAlgorithms.DQN,
                              self.dqn_trainer.algo, DefinedAlgorithms.DQN,
                              use_game_id=False, save_dir=save_dir + "cop_dqn_mrx_dqn")
        verbose_print(f"Simulations took {datetime.now() - now}", verbose=self.verbose)
        now = datetime.now()
        self.batch_simulation(self.number_of_test_games_per_pause,
                              self.ppo_trainer.algo, DefinedAlgorithms.PPO,
                              self.ppo_trainer.algo, DefinedAlgorithms.PPO,
                              use_game_id=False, save_dir=save_dir + "cop_ppo_mrx_ppo")
        verbose_print(f"Simulations took {datetime.now() - now}", verbose=self.verbose)
        now = datetime.now()
        self.batch_simulation(self.number_of_test_games_per_pause,
                              None, DefinedAlgorithms.RANDOM,
                              self.ppo_trainer.algo, DefinedAlgorithms.PPO,
                              use_game_id=False, save_dir=save_dir + "cop_random_mrx_ppo")
        verbose_print(f"Simulations took {datetime.now() - now}", verbose=self.verbose)
        now = datetime.now()
        self.batch_simulation(self.number_of_test_games_per_pause,
                              self.ppo_trainer.algo, DefinedAlgorithms.PPO,
                              None, DefinedAlgorithms.RANDOM,
                              use_game_id=False, save_dir=save_dir + "cop_ppo_mrx_random")
        verbose_print(f"Simulations took {datetime.now() - now}", verbose=self.verbose)
        now = datetime.now()
        self.batch_simulation(self.number_of_test_games_per_pause,
                              self.dqn_trainer.algo, DefinedAlgorithms.DQN,
                              None, DefinedAlgorithms.RANDOM,
                              use_game_id=False, save_dir=save_dir + "cop_dqn_mrx_random")
        verbose_print(f"Simulations took {datetime.now() - now}", verbose=self.verbose)
        now = datetime.now()
        self.batch_simulation(self.number_of_test_games_per_pause,
                              None, DefinedAlgorithms.RANDOM,
                              self.dqn_trainer.algo, DefinedAlgorithms.DQN,
                              use_game_id=False, save_dir=save_dir + "cop_random_mrx_dqn")
        verbose_print(f"Simulations took {datetime.now() - now}", verbose=self.verbose)
        now = datetime.now()
        self.batch_simulation(self.number_of_test_games_per_pause,
                              None, DefinedAlgorithms.RANDOM,
                              None, DefinedAlgorithms.RANDOM,
                              use_game_id=False, save_dir=save_dir + "cop_random_mrx_random")
        verbose_print(f"Simulations took {datetime.now() - now}", verbose=self.verbose)

    def run_train_experiment(self):
        """
        Description: This function runs the train experiment based on the configuration.
        :return: SimulationController - instance of the SimulationController
        """
        self.simulation_start_time = datetime.now()
        verbose_print("Running train experiment", self.verbose)

        current_train_iteration = 0
        verbose_print("Running simulations with 0 training", self.verbose)
        self.simulate_all_variants(self.save_dir + "/train_experiment/")
        self.merge_csv_train_experiment_results(current_train_iteration)
        while current_train_iteration < self.train_experiment_training_count:
            verbose_print(f"Training iteration {current_train_iteration + 1} of {self.train_experiment_training_count}",
                          self.verbose)
            self.last_simulation_time = datetime.now()
            self.dqn_trainer.train(number_of_iterations=1, save_interval=1)
            self.ppo_trainer.train(number_of_iterations=1, save_interval=1)
            verbose_print(
                f"Training iteration {current_train_iteration + 1} of {self.train_experiment_training_count} done in {datetime.now() - self.last_simulation_time}",
                self.verbose)
            current_train_iteration += 1
            if current_train_iteration % self.get_current_test_game_every_n_trainings(current_train_iteration) == 0:
                self.last_simulation_time = datetime.now()
                self.simulate_all_variants(self.save_dir + "/train_experiment/")
                verbose_print(f"Simulations took {datetime.now() - self.last_simulation_time}", self.verbose)
                self.last_simulation_time = datetime.now()
                self.merge_csv_train_experiment_results(current_train_iteration)
                verbose_print(f"Merge took {datetime.now() - self.last_simulation_time}", self.verbose)
                self.last_simulation_time = None

            if current_train_iteration % 50 == 0:
                from distutils.dir_util import copy_tree
                copy_tree("trained_models_dqn", "trained_models_dqn_copy")
                copy_tree("trained_models_ppo", "trained_models_ppo_copy")

        self.merge_final_train_experiment_results()
        verbose_print("Train experiment done", self.verbose)
        return self

    def cleanup(self):
        """
        Description: This function cleans up after the simulation.
        :return: None
        """
        ray.shutdown()
        for directory in self.experiment_directories:
            files = glob.glob(os.path.join(self.save_dir + "/train_experiment/" + directory + "/", 'game_*.csv'))
            files2 = glob.glob(os.path.join(self.save_dir + "/train_experiment/" + directory + "/", 'iteration_*.csv'))
            for file in files:
                os.remove(file)
            for file in files2:
                os.remove(file)

    def merge_final_train_experiment_results(self):
        """
        Description: This function merges the final results of the train experiment.
        :return: None
        """
        merged_data = pandas.DataFrame()
        for directory in self.experiment_directories:
            directory = self.save_dir + "/train_experiment/" + directory
            iteration_files = glob.glob(os.path.join(directory + "/", 'iteration_*.csv'))

            for file in iteration_files:
                data = pandas.read_csv(file)
                merged_data = pandas.concat([merged_data, data], ignore_index=False)

        merged_data.to_csv(self.save_dir + "/train_experiment/results.csv", index=False)

    def merge_csv_train_experiment_results(self, current_train_iteration):
        """
        Description: This function merges the results of the train experiment.
        :param current_train_iteration:  int - current training iteration
        :return: None
        """
        for directory in self.experiment_directories:
            merged_data = pandas.DataFrame()
            directory = self.save_dir + "/train_experiment/" + directory
            game_files = glob.glob(os.path.join(directory + "/", 'game_*.csv'))

            for file in game_files:
                data = pandas.read_csv(file)
                merged_data = pandas.concat([merged_data, data], ignore_index=True)

            merged_data["train_iteration"] = current_train_iteration
            merged_data.to_csv(directory + f"/iteration_{current_train_iteration}.csv", index=False)
            game_files = glob.glob(os.path.join(directory + "/", 'game_*.csv'))
            for file in game_files:
                os.remove(file)

    def merge_csv_results(self, sim_number):
        """
        Description: This function merges the results of the simulation.
        :param sim_number: int - current simulation number
        :return: None
        """
        game_files = glob.glob(os.path.join(self.save_dir + "/", 'game_*.csv'))
        merged_data = pandas.DataFrame()

        for file in game_files:
            data = pandas.read_csv(file)
            merged_data = pandas.concat([merged_data, data], ignore_index=True)

        # Save the merged DataFrame to a new CSV file
        merged_data.to_csv(f"{self.save_dir}/simulation_{sim_number}", index=False)

    def merge_csv_simulation_experiment_results(self):
        """
        Description: This function merges the results of the simulation experiment.
        :return: None
        """
        merged_data = pandas.DataFrame()
        for directory in self.experiment_directories:
            directory = self.simulation_experiment_dir + "/" + directory
            game_files = glob.glob(os.path.join(directory + "/", 'game_*.csv'))

            for file in game_files:
                data = pandas.read_csv(file)
                merged_data = pandas.concat([merged_data, data], ignore_index=True)

            game_files = glob.glob(os.path.join(directory + "/", 'game_*.csv'))
            for file in game_files:
                os.remove(file)
        merged_data.to_csv(self.simulation_experiment_dir + "/results.csv", index=False)

    def simulate_game(self, cop_algo, cop_algo_selected: DefinedAlgorithms, mr_x_algo,
                      mr_x_algo_selected: DefinedAlgorithms, save_dir=None):
        """
        Description: This function simulates a single game based on the configuration.
        :param cop_algo: Trainer - instance of the cop agent
        :param cop_algo_selected: DefinedAlgorithms - selected algorithm for the cop agent
        :param mr_x_algo: Trainer - instance of the mr_x agent
        :param mr_x_algo_selected: DefinedAlgorithms - selected algorithm for the mr_x agent
        :param save_dir: str - directory to save the results
        :return: None
        """
        self.game.reset()
        turn_stats = {}
        while self.game.get_game_status() is scotland_yard_game_logic.GameStatus.ONGOING:
            self.game.play_turn(cop_algo=cop_algo, mr_x_algo=mr_x_algo, verbose=True,
                                train_simulation=not self.run_simulation_experiment)
            if self.game.playing_player_index == len(self.game.players) - 1:
                if self.game.get_game_status() is scotland_yard_game_logic.GameStatus.ONGOING:
                    turn_stats[self.game.get_current_round_number()] = self.get_round_statistics()
        turn_stats[self.game.get_current_round_number()] = self.get_round_statistics()
        game_stats = self.get_game_statistics()
        game_stats["game_id"] = self.current_game_id
        game_stats["mr_x_algo"] = mr_x_algo_selected.value
        game_stats["cop_algo"] = cop_algo_selected.value
        self.save_statistics(game_stats, turn_stats, save_dir)

    def save_statistics(self, game_stats, turn_stats, save_dir):
        """
        Description: This function saves the statistics of the game.
        :param game_stats: dict - game statistics
        :param turn_stats: dict - turn statistics
        :param save_dir: str - directory to save the results
        :return: None
        """
        if save_dir is None:
            save_dir = self.save_dir
        file_name = save_dir + "/game_" + str(self.current_game_id) + ".csv"

        rounds_df = pandas.DataFrame.from_dict(turn_stats, orient='index')
        game_stats["mr_x_avg_distance_to_cop"] = rounds_df["mr_x_avg_distance_to_cop"].mean()
        game_stats["avg_distance_between_cops"] = rounds_df["avg_distance_between_cops"].mean()
        game_stats["mr_x_avg_reward"] = rounds_df["mr_x_reward"].mean()
        game_stats["cops_avg_reward"] = rounds_df["cops_avg_reward"].mean()
        result_df = pandas.DataFrame([game_stats])

        result_df.to_csv(file_name, index=False)

    def get_round_statistics(self):
        """
        Description: This function returns the statistics of the round.
        :return: dict - round statistics
        """
        mr_x_avg_distance_to_cop = 0
        avg_distance_between_cops = 0

        for cop in self.game.get_cops():
            mr_x_avg_distance_to_cop += self.game.get_mr_x().get_distance_to(cop.get_position())
            cop_distance_to_other_cop = 0
            for other_cop in self.game.get_cops():
                cop_distance_to_other_cop = cop.get_distance_to(other_cop.get_position())
            avg_distance_between_cops += cop_distance_to_other_cop / 2

        mr_x_avg_distance_to_cop /= len(self.game.get_cops())
        rewards = self.game.get_simulation_rewards()
        return {"mr_x_avg_distance_to_cop": mr_x_avg_distance_to_cop,
                "avg_distance_between_cops": avg_distance_between_cops,
                "mr_x_reward": rewards["mr_x"],
                "cops_avg_reward": sum([rewards[cop.name] for cop in self.game.get_cops()]) / len(self.game.get_cops())}

    def run_simulation_experiment(self):
        """
        Description: This function runs the simulation experiment.
        :return: None
        """
        self.number_of_test_games_per_pause = 10000
        self.simulate_all_variants(self.simulation_experiment_dir)
        self.merge_csv_simulation_experiment_results()

    def get_game_statistics(self):
        """
        Description: This function returns the statistics of the game.
        """
        game_result = self.game.get_game_status().value
        return {"game_result": game_result}
