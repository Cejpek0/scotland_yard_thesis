import os

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

from src.game.scotland_yard_game_logic import DefinedAlgorithms, GameStatus


class SimulationProcessor:
    def __init__(self):
        #relevant cols: cop_algo, mr_x_algo, game_result, game_id, mr_x_avg_distance_to_cop, avg_distance_between_cops, mr_x_reward,cops_avg_reward
        self.graphs_dir = "../simulations/graphs"
        self.experiment_dir = "../simulations/train_experiment"
        self.dataframe = self.load_experiment()
        sns.set_style()
        if not os.path.exists(self.graphs_dir):
            os.makedirs(self.graphs_dir)

    def generate_graphs(self):
        pass

    def get_dataframe_for(self, cop_selected_algo: DefinedAlgorithms,
                          mr_x_selected_algo: DefinedAlgorithms) -> DataFrame:
        new_dataframe = self.dataframe[
            (self.dataframe["cop_algo"] == cop_selected_algo.value) & (
                    self.dataframe["mr_x_algo"] == mr_x_selected_algo.value)]
        return new_dataframe

    def generate_graph_for(self, dataframe: DataFrame, cop_selected_algo: DefinedAlgorithms,
                           mr_x_selected_algo: DefinedAlgorithms):
        label_mapping = {
            'mr_x_avg_distance_to_cop': 'Vzdálenost mezi Panem X a policisty',
            'mr_x_avg_reward': 'Odměna Pana X',
            'cops_avg_reward': 'Odměna policistů',
        }
        # Graph 1: Result (scatter), Distances (lines)
        g = sns.relplot(
            x='train_iteration',
            y='value',
            hue='variable',
            style='variable',
            kind='line',
            errorbar=("pi", 98),
            facet_kws=dict(legend_out=False),
            data=dataframe.melt(id_vars='train_iteration',
                                value_vars=['mr_x_avg_reward', 'cops_avg_reward',
                                            'mr_x_avg_distance_to_cop']),
        )
        for text in g.legend.texts:
            if text.get_text() in label_mapping.keys():
                text.set_text(label_mapping[text.get_text()])
        g.legend.set_title("Sledované hodnoty")
        g.set_ylabels("Hodnota")
        g.set_xlabels("Počet trénovacích iterací")
        plt.title(f"Policisté - {cop_selected_algo.name}\nPan X - {mr_x_selected_algo.name}")

        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        ax = g.ax
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth(1)
        if not os.path.exists(self.graphs_dir):
            os.makedirs(self.graphs_dir)
        g.savefig(f"{self.graphs_dir}/cop_{cop_selected_algo.name}_mrx_{mr_x_selected_algo.name}", bbox_inches="tight")

    def load_experiment(self) -> DataFrame:
        assert os.path.exists(self.experiment_dir + "/results.csv"), "Experiment results file not found"
        return pd.read_csv(self.experiment_dir + "/results.csv")

    def print_results(self):
        dataframe_cop_ppo_mr_x_ppo = self.get_dataframe_for(DefinedAlgorithms.PPO, DefinedAlgorithms.PPO)
        dataframe_cop_ppo_mr_x_random = self.get_dataframe_for(DefinedAlgorithms.PPO, DefinedAlgorithms.RANDOM)
        dataframe_cop_ppo_mr_x_dqn = self.get_dataframe_for(DefinedAlgorithms.PPO, DefinedAlgorithms.DQN)
        dataframe_cop_random_mr_x_ppo = self.get_dataframe_for(DefinedAlgorithms.RANDOM, DefinedAlgorithms.PPO)
        dataframe_cop_random_mr_x_random = self.get_dataframe_for(DefinedAlgorithms.RANDOM, DefinedAlgorithms.RANDOM)
        dataframe_cop_random_mr_x_dqn = self.get_dataframe_for(DefinedAlgorithms.RANDOM, DefinedAlgorithms.DQN)
        dataframe_cop_dqn_mr_x_ppo = self.get_dataframe_for(DefinedAlgorithms.DQN, DefinedAlgorithms.PPO)
        dataframe_cop_dqn_mr_x_random = self.get_dataframe_for(DefinedAlgorithms.DQN, DefinedAlgorithms.RANDOM)
        dataframe_cop_dqn_mr_x_dqn = self.get_dataframe_for(DefinedAlgorithms.DQN, DefinedAlgorithms.DQN)

        number_of_victories_ppo_against_random = len(
            dataframe_cop_ppo_mr_x_random[dataframe_cop_ppo_mr_x_random["game_result"] == GameStatus.COPS_WON.value])
        number_of_victories_ppo_against_random += len(
            dataframe_cop_random_mr_x_ppo[dataframe_cop_random_mr_x_ppo["game_result"] == GameStatus.MR_X_WON.value])
        total_games_ppo_against_random = len(dataframe_cop_ppo_mr_x_random) + len(dataframe_cop_random_mr_x_ppo)

        number_of_victories_ppo_against_dqn = len(
            dataframe_cop_ppo_mr_x_dqn[dataframe_cop_ppo_mr_x_dqn["game_result"] == GameStatus.COPS_WON.value])
        number_of_victories_ppo_against_dqn += len(
            dataframe_cop_dqn_mr_x_ppo[dataframe_cop_dqn_mr_x_ppo["game_result"] == GameStatus.MR_X_WON.value])
        total_games_ppo_against_dqn = len(dataframe_cop_ppo_mr_x_dqn) + len(dataframe_cop_dqn_mr_x_ppo)

        number_of_victories_dqn_against_random = len(
            dataframe_cop_dqn_mr_x_random[dataframe_cop_dqn_mr_x_random["game_result"] == GameStatus.COPS_WON.value])
        number_of_victories_dqn_against_random += len(
            dataframe_cop_random_mr_x_dqn[dataframe_cop_random_mr_x_dqn["game_result"] == GameStatus.MR_X_WON.value])
        total_games_dqn_against_random = len(dataframe_cop_dqn_mr_x_random) + len(dataframe_cop_random_mr_x_dqn)

        ppo_avg_distance_to_cop = dataframe_cop_ppo_mr_x_ppo["mr_x_avg_distance_to_cop"].mean()
        random_avg_distance_to_cop = dataframe_cop_random_mr_x_random["mr_x_avg_distance_to_cop"].mean()
        dqn_avg_distance_to_cop = dataframe_cop_dqn_mr_x_dqn["mr_x_avg_distance_to_cop"].mean()

        text = f"""Výsledky simulace:
        Počet výher PPO proti Random: {number_of_victories_ppo_against_random}/{total_games_ppo_against_random}
        Počet výher PPO proti DQN: {number_of_victories_ppo_against_dqn}/{total_games_ppo_against_dqn}
        Počet výher DQN proti Random: {number_of_victories_dqn_against_random}/{total_games_dqn_against_random}
        Průměrná vzdálenost mezi Panem X a policisty (PPO): {ppo_avg_distance_to_cop}
        Průměrná vzdálenost mezi Panem X a policisty (Random): {random_avg_distance_to_cop}
        Průměrná vzdálenost mezi Panem X a policisty (DQN): {dqn_avg_distance_to_cop}
        """

        file = open(f"{self.graphs_dir}/results.txt", "w", encoding="utf-8")
        file.write(text)
        file.close()
        
        print(text)


if __name__ == '__main__':
    simulation = SimulationProcessor()
    dataframe = simulation.get_dataframe_for(DefinedAlgorithms.PPO, DefinedAlgorithms.PPO)
    simulation.generate_graph_for(dataframe, DefinedAlgorithms.PPO, DefinedAlgorithms.PPO)

    dataframe = simulation.get_dataframe_for(DefinedAlgorithms.PPO, DefinedAlgorithms.RANDOM)
    simulation.generate_graph_for(dataframe, DefinedAlgorithms.PPO, DefinedAlgorithms.RANDOM)

    dataframe = simulation.get_dataframe_for(DefinedAlgorithms.RANDOM, DefinedAlgorithms.PPO)
    simulation.generate_graph_for(dataframe, DefinedAlgorithms.RANDOM, DefinedAlgorithms.PPO)

    dataframe = simulation.get_dataframe_for(DefinedAlgorithms.PPO, DefinedAlgorithms.DQN)
    simulation.generate_graph_for(dataframe, DefinedAlgorithms.PPO, DefinedAlgorithms.DQN)

    dataframe = simulation.get_dataframe_for(DefinedAlgorithms.DQN, DefinedAlgorithms.PPO)
    simulation.generate_graph_for(dataframe, DefinedAlgorithms.DQN, DefinedAlgorithms.PPO)

    dataframe = simulation.get_dataframe_for(DefinedAlgorithms.DQN, DefinedAlgorithms.DQN)
    simulation.generate_graph_for(dataframe, DefinedAlgorithms.DQN, DefinedAlgorithms.DQN)

    dataframe = simulation.get_dataframe_for(DefinedAlgorithms.DQN, DefinedAlgorithms.RANDOM)
    simulation.generate_graph_for(dataframe, DefinedAlgorithms.DQN, DefinedAlgorithms.RANDOM)

    dataframe = simulation.get_dataframe_for(DefinedAlgorithms.RANDOM, DefinedAlgorithms.DQN)
    simulation.generate_graph_for(dataframe, DefinedAlgorithms.RANDOM, DefinedAlgorithms.DQN)

    dataframe = simulation.get_dataframe_for(DefinedAlgorithms.RANDOM, DefinedAlgorithms.RANDOM)
    simulation.generate_graph_for(dataframe, DefinedAlgorithms.RANDOM, DefinedAlgorithms.RANDOM)

    simulation.print_results()
