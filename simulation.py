from src.simulations.SimulationController import SimulationController

if __name__ == '__main__':
    simulation = SimulationController(save_dir="simulations")
    config = {
        "turns_ppo_vs_ppo": 1,
        "turns_random_vs_ppo": 1,
        "turns_ppo_vs_random": 1,
        "turns_random_vs_random": 1
    }
    simulation.run(config)
