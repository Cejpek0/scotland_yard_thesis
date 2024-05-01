import os

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

from src.game.scotland_yard_game_logic import DefinedAlgorithms


class SimulationProcessor:
    def __init__(self):
        #relevant cols: cop_algo, mr_x_algo, game_result, game_id, mr_x_avg_distance_to_cop, avg_distance_between_cops, mr_x_reward,cops_avg_reward
        self.sim_dir = "../simulations/graphs"
        self.experiment_dir = "../simulations/train_experiment"
        self.dataframe = self.load_experiment()
        sns.set_style()

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
        y_vars = ['mr_x_avg_distance_to_cop', 'avg_distance_between_cops', 'mr_x_reward', 'cops_avg_reward']

        def add_index_col(df):
            df['temp_index'] = df.groupby('game_id').cumcount()
            return df

        dataframe = dataframe.groupby('game_id').apply(add_index_col)

        g = sns.FacetGrid(dataframe, col='game_id')
        g.map_dataframe(sns.lineplot, x='temp_index')
        g.add_legend()

        g.set_axis_labels("Index within Game", "Value")  # Adjust labels if needed 
        g.fig.suptitle(f'Metrics for Cop Algo: {cop_selected_algo}, Mr X Algo: {mr_x_selected_algo}')
        g.set_titles(col_template='{col_name}')
        plt.tight_layout()
        
    def load_experiment(self) -> DataFrame:
        assert os.path.exists(self.experiment_dir + "/results.csv"), "Experiment results file not found"
        return pd.read_csv(self.experiment_dir + "/results.csv")


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
