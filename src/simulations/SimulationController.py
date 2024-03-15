import csv
import random

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

        for i in range(turns_ppo_vs_ppo):
            self.simulate_game(scotland_yard_game_logic.DefinedAlgorithms.PPO,
                               scotland_yard_game_logic.DefinedAlgorithms.PPO)
            self.current_game_id += 1
        print("PPO vs PPO done")
        for i in range(turns_random_vs_ppo):
            self.simulate_game(scotland_yard_game_logic.DefinedAlgorithms.RANDOM,
                               scotland_yard_game_logic.DefinedAlgorithms.PPO)
            self.current_game_id += 1
        print("Random vs PPO done")
        for i in range(turns_ppo_vs_random):
            self.simulate_game(scotland_yard_game_logic.DefinedAlgorithms.PPO,
                               scotland_yard_game_logic.DefinedAlgorithms.RANDOM)
            self.current_game_id += 1
        print("PPO vs Random done")
        for i in range(turns_random_vs_random):
            self.simulate_game(scotland_yard_game_logic.DefinedAlgorithms.RANDOM,
                               scotland_yard_game_logic.DefinedAlgorithms.RANDOM)
            self.current_game_id += 1
        print("Random vs Random done")
        self.merge_csv_results(f"results{random.randint(0, 10000)}.csv")
    
    def merge_csv_results(self, file_name):
        pass
        

    def simulate_game(self, cop_agent, mr_x_agent):
        self.game.reset()
        rounds_stats = {}
        while self.game.get_game_status() is scotland_yard_game_logic.GameStatus.ONGOING:
            self.game.play_turn(cop_agent, mr_x_agent)
            if self.game.playing_player_index == len(self.game.players) - 1:
                rounds_stats[self.game.get_current_round_number()] = self.get_round_statistics()

        game_stats = self.get_game_statistics()
        game_stats["game_id"] = self.current_game_id
        game_stats["mr_x_agent"] = mr_x_agent.value
        game_stats["cop_agent"] = cop_agent.value
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
