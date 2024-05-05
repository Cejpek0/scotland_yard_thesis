"""
File description: This file contains the SimulationProcessor class, which is responsible for processing the results of the
experiments and generating graphs and text output based on the results.

Author: Michal Cejpek (xcejpe05@stud.fit.vutbr.cz)
"""
import os

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

from src.game.scotland_yard_game_logic import DefinedAlgorithms, GameStatus


class SimulationProcessor:
    def __init__(self, train_simulation):
        #relevant cols: cop_algo, mr_x_algo, game_result, game_id, mr_x_avg_distance_to_cop, avg_distance_between_cops, mr_x_reward,cops_avg_reward
        self.graphs_dir = "../simulations/graphs"
        self.train_experiment_dir = "../simulations/train_experiment"
        self.simulation_experiment_dir = "../simulations/simulation_experiment/"
        if train_simulation:
            self.dataframe = self.load_train_experiment()
        else:
            self.dataframe = self.load_simulation_experiment()
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
            'mr_x_avg_distance_to_cop': 'Vzdálenost policistů a Pana X',
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
            lw=2,
            errorbar=None,
            facet_kws=dict(legend_out=False),
            data=dataframe.melt(id_vars='train_iteration',
                                value_vars=['mr_x_avg_reward', 'cops_avg_reward',
                                            'mr_x_avg_distance_to_cop']),
        )
        for text in g.legend.texts:
            if text.get_text() in label_mapping.keys():
                text.set_text(label_mapping[text.get_text()])
        g.legend.set_title(None)
        g.set_ylabels("Hodnota")
        g.set_xlabels("Počet trénovacích iterací")
        plt.title(f"Policisté - {cop_selected_algo.name}\nPan X - {mr_x_selected_algo.name}")

        sns.move_legend(g, "lower center")
        plt.setp(g._legend.get_texts(), fontsize='8')
        ax = g.ax
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth(1)
        plt.subplots_adjust(bottom=0.25)
        if not os.path.exists(self.graphs_dir):
            os.makedirs(self.graphs_dir)
        g.savefig(f"{self.graphs_dir}/cop_{cop_selected_algo.name}_mrx_{mr_x_selected_algo.name}", bbox_inches="tight")

    def load_train_experiment(self) -> DataFrame:
        assert os.path.exists(self.train_experiment_dir + "/results.csv"), "Experiment results file not found"
        return pd.read_csv(self.train_experiment_dir + "/results.csv")

    def load_simulation_experiment(self) -> DataFrame:
        assert os.path.exists(self.simulation_experiment_dir), "Experiment results file not found"
        return pd.read_csv(self.simulation_experiment_dir + "results.csv")

    def print_results(self, train_simulation):
        dataframe_cop_ppo_mr_x_ppo = self.get_dataframe_for(DefinedAlgorithms.PPO, DefinedAlgorithms.PPO)
        dataframe_cop_ppo_mr_x_random = self.get_dataframe_for(DefinedAlgorithms.PPO, DefinedAlgorithms.RANDOM)
        dataframe_cop_ppo_mr_x_dqn = self.get_dataframe_for(DefinedAlgorithms.PPO, DefinedAlgorithms.DQN)
        dataframe_cop_random_mr_x_ppo = self.get_dataframe_for(DefinedAlgorithms.RANDOM, DefinedAlgorithms.PPO)
        dataframe_cop_random_mr_x_random = self.get_dataframe_for(DefinedAlgorithms.RANDOM, DefinedAlgorithms.RANDOM)
        dataframe_cop_random_mr_x_dqn = self.get_dataframe_for(DefinedAlgorithms.RANDOM, DefinedAlgorithms.DQN)
        dataframe_cop_dqn_mr_x_ppo = self.get_dataframe_for(DefinedAlgorithms.DQN, DefinedAlgorithms.PPO)
        dataframe_cop_dqn_mr_x_random = self.get_dataframe_for(DefinedAlgorithms.DQN, DefinedAlgorithms.RANDOM)
        dataframe_cop_dqn_mr_x_dqn = self.get_dataframe_for(DefinedAlgorithms.DQN, DefinedAlgorithms.DQN)

        victories_ppo_cop_vs_mrx_random = len(
            dataframe_cop_ppo_mr_x_random[dataframe_cop_ppo_mr_x_random["game_result"] == GameStatus.COPS_WON.value])
        games_ppo_cop_vs_mrx_random = len(dataframe_cop_ppo_mr_x_random)
        victories_ppo_mrx_vs_cop_random = len(
            dataframe_cop_random_mr_x_ppo[dataframe_cop_random_mr_x_ppo["game_result"] == GameStatus.MR_X_WON.value])
        games_ppo_mrx_vs_cop_random = len(dataframe_cop_random_mr_x_ppo)
        total_games_ppo_vs_random = len(dataframe_cop_ppo_mr_x_random) + len(dataframe_cop_random_mr_x_ppo)
        victories_ppo_vs_random = victories_ppo_cop_vs_mrx_random + victories_ppo_mrx_vs_cop_random

        victories_ppo_cop_vs_mrx_ppo = len(
            dataframe_cop_ppo_mr_x_ppo[dataframe_cop_ppo_mr_x_ppo["game_result"] == GameStatus.COPS_WON.value])
        victories_ppo_mrx_vs_cop_ppo = len(
            dataframe_cop_ppo_mr_x_ppo[dataframe_cop_ppo_mr_x_ppo["game_result"] == GameStatus.MR_X_WON.value])
        total_games_ppo_vs_ppo = len(dataframe_cop_ppo_mr_x_ppo) + len(dataframe_cop_ppo_mr_x_ppo)

        if train_simulation:
            random_ppo_early_wins_mrx = len(dataframe_cop_ppo_mr_x_random[
                                                (dataframe_cop_ppo_mr_x_random['train_iteration'] < 100) & (
                                                        dataframe_cop_ppo_mr_x_random[
                                                            'game_result'] == GameStatus.MR_X_WON.value)])
            random_ppo_early_wins_cops = len(dataframe_cop_random_mr_x_ppo[
                                                 (dataframe_cop_random_mr_x_ppo['train_iteration'] < 100) & (
                                                         dataframe_cop_random_mr_x_ppo[
                                                             'game_result'] == GameStatus.COPS_WON.value)])

        victories_ppo_cop_vs_mrx_dqn = len(
            dataframe_cop_ppo_mr_x_dqn[dataframe_cop_ppo_mr_x_dqn["game_result"] == GameStatus.COPS_WON.value])
        games_ppo_cop_vs_mrx_dqn = len(dataframe_cop_ppo_mr_x_dqn)
        victories_ppo_mrx_vs_cop_dqn = len(
            dataframe_cop_dqn_mr_x_ppo[dataframe_cop_dqn_mr_x_ppo["game_result"] == GameStatus.MR_X_WON.value])
        games_ppo_mrx_vs_cop_dqn = len(dataframe_cop_dqn_mr_x_ppo)
        total_games_ppo_vs_dqn = len(dataframe_cop_ppo_mr_x_dqn) + len(dataframe_cop_dqn_mr_x_ppo)
        victories_ppo_vs_dqn = victories_ppo_cop_vs_mrx_dqn + victories_ppo_mrx_vs_cop_dqn

        victories_dqn_cop_vs_mrx_random = len(
            dataframe_cop_dqn_mr_x_random[dataframe_cop_dqn_mr_x_random["game_result"] == GameStatus.COPS_WON.value])
        games_dqn_cop_vs_mrx_random = len(dataframe_cop_dqn_mr_x_random)
        victories_dqn_mrx_vs_cop_random = len(
            dataframe_cop_random_mr_x_dqn[dataframe_cop_random_mr_x_dqn["game_result"] == GameStatus.MR_X_WON.value])
        games_dqn_mrx_vs_cop_random = len(dataframe_cop_random_mr_x_dqn)
        total_games_dqn_vs_random = len(dataframe_cop_dqn_mr_x_random) + len(dataframe_cop_random_mr_x_dqn)
        victories_dqn_vs_random = victories_dqn_cop_vs_mrx_random + victories_dqn_mrx_vs_cop_random

        victories_dqn_cop_vs_mrx_dqn = len(
            dataframe_cop_dqn_mr_x_dqn[dataframe_cop_dqn_mr_x_dqn["game_result"] == GameStatus.COPS_WON.value])
        total_games_dqn_vs_random = len(dataframe_cop_dqn_mr_x_dqn) + len(dataframe_cop_dqn_mr_x_dqn)
        victories_dqn_mrx_vs_cop_dqn = len(
            dataframe_cop_dqn_mr_x_dqn[dataframe_cop_dqn_mr_x_dqn["game_result"] == GameStatus.MR_X_WON.value])

        random_vs_random_cop_wins = len(
            dataframe_cop_random_mr_x_random[
                dataframe_cop_random_mr_x_random["game_result"] == GameStatus.COPS_WON.value])
        total_games_random_vs_random = len(dataframe_cop_random_mr_x_random)

        ppo_avg_distance_to_cop = dataframe_cop_ppo_mr_x_ppo["mr_x_avg_distance_to_cop"].mean()
        random_avg_distance_to_cop = dataframe_cop_random_mr_x_random["mr_x_avg_distance_to_cop"].mean()
        dqn_avg_distance_to_cop = dataframe_cop_dqn_mr_x_dqn["mr_x_avg_distance_to_cop"].mean()

        string_early_wins = ""
        if train_simulation:
            string_early_wins = f"Počet výher náhodných agentů proti PPO agentům (do 100 iterace): {random_ppo_early_wins_mrx + random_ppo_early_wins_cops}/{total_games_ppo_vs_random - victories_ppo_vs_random}"
            string_early_wins += f"\nZ toho počet výher Pana X: {random_ppo_early_wins_mrx}/{random_ppo_early_wins_mrx + random_ppo_early_wins_cops}"
            string_early_wins += f"\nZ toho počet výher policistů: {random_ppo_early_wins_cops}/{random_ppo_early_wins_mrx + random_ppo_early_wins_cops}"
        text = f"""Výsledky simulace:
        Počet výher PPO policistů proti náhodnému Panu X: {victories_ppo_cop_vs_mrx_random}/{games_ppo_cop_vs_mrx_random}
        Počet výher náhodného Pana X proti PPO policistům: {games_ppo_cop_vs_mrx_random - victories_ppo_cop_vs_mrx_random}/{games_ppo_cop_vs_mrx_random}
        Počet výher PPO Panu X proti náhodným policistům: {victories_ppo_mrx_vs_cop_random}/{games_ppo_mrx_vs_cop_random}
        Počet výher náhodných policistů proti PPO Panu X: {games_ppo_mrx_vs_cop_random - victories_ppo_mrx_vs_cop_random}/{games_ppo_mrx_vs_cop_random}
        Počet výher PPO proti náhodnému agentovi: {victories_ppo_vs_random}/{total_games_ppo_vs_random}
        {string_early_wins}
        
        Počet výher PPO policistů proti PPO Panu X: {victories_ppo_cop_vs_mrx_ppo}/{total_games_ppo_vs_ppo}
        Počet výher PPO Panu X proti PPO policistům: {victories_ppo_mrx_vs_cop_ppo}/{total_games_ppo_vs_ppo}
        
        Počet výher PPO policistů proti DQN Panu X: {victories_ppo_cop_vs_mrx_dqn}/{games_ppo_cop_vs_mrx_dqn}
        Počet výher DQN Panu X proti PPO policistům: {games_ppo_cop_vs_mrx_dqn - victories_ppo_cop_vs_mrx_dqn}/{games_ppo_cop_vs_mrx_dqn}
        Počet výher PPO Panu X proti DQN policistům: {victories_ppo_mrx_vs_cop_dqn}/{total_games_ppo_vs_dqn}
        Počet výher DQN policistů proti PPO Panu X: {games_ppo_mrx_vs_cop_dqn - victories_ppo_mrx_vs_cop_dqn}/{total_games_ppo_vs_dqn}
        Počet výher PPO proti DQN agentovi: {victories_ppo_vs_dqn}/{total_games_ppo_vs_dqn}
        
        Počet výher DQN policistů proti náhodnému Panu X: {victories_dqn_cop_vs_mrx_random}/{games_dqn_cop_vs_mrx_random} 
        Počet výher náhodného Pana X proti DQN policistům: {games_dqn_cop_vs_mrx_random - victories_dqn_cop_vs_mrx_random}/{games_dqn_cop_vs_mrx_random}
        Počet výher DQN Panu X proti náhodným policistům: {victories_dqn_mrx_vs_cop_random}/{games_dqn_mrx_vs_cop_random}
        Počet výher náhodných policistů proti DQN Panu X: {games_dqn_mrx_vs_cop_random - victories_dqn_mrx_vs_cop_random}/{games_dqn_mrx_vs_cop_random}
        Počet výher DQN proti náhodnému agentovi: {victories_dqn_vs_random}/{total_games_dqn_vs_random}
        
        Počet výher DQN policistů proti DQN Panu X: {victories_dqn_cop_vs_mrx_dqn}/{total_games_dqn_vs_random}
        Počet výher DQN Panu X proti DQN policistům: {victories_dqn_mrx_vs_cop_dqn}/{total_games_dqn_vs_random}
        
        Počet výher náhodných policistů proti náhodnému Panu X: {random_vs_random_cop_wins}/{total_games_random_vs_random}
        
        Průměrná vzdálenost mezi Panem X a policisty (PPO): {ppo_avg_distance_to_cop}
        Průměrná vzdálenost mezi Panem X a policisty (náhodný): {random_avg_distance_to_cop}
        Průměrná vzdálenost mezi Panem X a policisty (DQN): {dqn_avg_distance_to_cop}
        """

        directory = self.train_experiment_dir + "/" if train_simulation else self.simulation_experiment_dir
        directory += "results.txt"

        file = open(f"{directory}", "w", encoding="utf-8")
        file.write(text)
        file.close()

        print(text)


if __name__ == '__main__':
    simulation = SimulationProcessor(train_simulation=True)
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

    simulation.print_results(True)
