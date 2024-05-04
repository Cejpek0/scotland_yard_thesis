from src.SimulationController import SimulationController

if __name__ == '__main__':
    simulation = SimulationController(save_dir="simulations",
                                      verbose=True,
                                      simulation_experiment=True)
    simulation.run_simulation_experiment()
    #simulation.run_train_experiment().cleanup()
