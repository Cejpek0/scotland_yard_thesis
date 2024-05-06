"""
File description: This file is the main entry point of simulations and experiments.
It creates an instance of the SimulationController and runs defined experiment simulation.

Author: Michal Cejpek (xcejpe05@stud.fit.vutbr.cz)
"""
from src.SimulationController import SimulationController

if __name__ == '__main__':
    simulation = SimulationController(save_dir="simulations",
                                      verbose=True,
                                      simulation_experiment=False)
    # simulation.run_simulation_experiment()
    simulation.run_train_experiment().cleanup()
