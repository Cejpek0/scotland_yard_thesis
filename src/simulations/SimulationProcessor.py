import glob
import os

import pandas as pd


from src.game.scotland_yard_game_logic import DefinedAlgorithms


class SimulationController:
    def __init__(self, save_dir):
        self.sim_dir = save_dir

    def run(self, config):
        csv_files = glob.glob(os.path.join(self.sim_dir, '*.csv'))
        dataframe = pd.concat([pd.read_csv(file) for file in csv_files])
        
        
            
