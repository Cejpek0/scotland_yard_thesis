from src.SimulationController import SimulationController

if __name__ == '__main__':
    simulation = SimulationController(save_dir="simulations",
                                      verbose=True,
                                      experiment_training_iteration_count=2000,
                                      test_games_every_n_trainings=10,
                                      test_games_count_per_pause=100)
    config = {
        "turns_ppo_vs_ppo": 1,
        "turns_random_vs_ppo": 1,
        "turns_ppo_vs_random": 1,
        "turns_random_vs_random": 1
    }
    simulation.run_train_experiment().cleanup()
