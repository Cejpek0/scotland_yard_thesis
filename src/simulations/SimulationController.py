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
        ray.init(num_gpus=1)
    
    
        def env_creator(env_config):
            return ScotlandYardEnvironment({}, scotland_yard_game.DefinedAlgorithms.PPO)
    
    
        register_env("scotland_env", env_creator)
    
        ModelCatalog.register_custom_model(
            "cc_model",
            TorchCentralizedCriticModel
        )
    
        my_config = (PPOConfig()
                     .training(model={"custom_model": "cc_model"}))
    
        my_config["policies"] = {
            "mr_x_policy": scotland_yard_game.MR_X_POLICY_SPEC,
            "cop_policy": scotland_yard_game.COP_POLICY_SPEC,
        }
    
    
        def policy_mapping_fn(agent_id, episode, worker):
            return "mr_x_policy" if agent_id == "mr_x" else "cop_policy"
    
    
        my_config["policy_mapping_fn"] = policy_mapping_fn
        repeat = 20
        my_config["num_iterations"] = repeat
        my_config["num_rollout_workers"] = 4
        my_config["reuse_actors"] = True
        my_config.resources(num_gpus=1, num_gpus_per_worker=0.2)
        my_config.framework("torch")
    
        # Set the config object's env.
        algo = PPO(env="scotland_env", config=my_config)
    
    
        # check if trained policies exist
        directory = "trained_policies"
    
        if os.path.exists(directory):
            print("Loading policies")
            algo.restore(directory)
        for i in range(repeat):
            print("Training iteration {} of {}".format(i + 1, repeat))
            print(algo.train())
            if i % 4 == 0:
                print("Saving policies")
                algo.save(directory)
        algo.save(directory)
        ray.shutdown()
        pass

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
