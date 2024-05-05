"""
File description: Game scene class for Scotland Yard game.
This class is responsible for communication between game and gui.

Author: Michal Cejpek (xcejpe05@stud.fit.vutbr.cz)
"""
import time

import pygame.time

from src.GameController import GameController
from src.GuiController import GuiController
from src.game.scotland_yard_game_logic import ScotlandYardGameLogic, GameStatus, DefinedAlgorithms
from src.game.scotland_yard_game_visual import ScotlandYardGameVisual, GRID_SIZE
from src.helper import verbose_print
from src.scenes.scene import Scene


# Constants

class ScotlandYardScene(Scene):
    def __init__(self, game_controller: GameController, gui_controller: GuiController,
                 cop_selected_algorithm: DefinedAlgorithms,
                 mr_x_selected_algorithm: DefinedAlgorithms):
        Scene.__init__(self, game_controller, gui_controller)
        self.time_of_end = None
        self.game = ScotlandYardGameLogic(verbose=self.game_controller.verbose)
        self.game_visual = ScotlandYardGameVisual(self.game, gui_controller)
        self.cell_size = gui_controller.width // GRID_SIZE
        if mr_x_selected_algorithm is DefinedAlgorithms.PPO:
            self.mrx_algorithm = self.game_controller.algo_ppo
        elif mr_x_selected_algorithm is DefinedAlgorithms.DQN:
            self.mrx_algorithm = self.game_controller.algo_dqn
        else:
            self.mrx_algorithm = None
        if cop_selected_algorithm is DefinedAlgorithms.PPO:
            self.cop_algorithm = self.game_controller.algo_ppo
        elif cop_selected_algorithm is DefinedAlgorithms.DQN:
            self.cop_algorithm = self.game_controller.algo_dqn
        else:
            self.cop_algorithm = None

    def update(self, user_input):
        from src.GameController import UserActions

        if user_input[UserActions.space.name] and self.game.get_game_status() == GameStatus.ONGOING:
            self.game_controller.playing = not self.game_controller.playing
        elif self.game_controller.playing:
            self.game.play_turn(cop_algo=self.cop_algorithm, mr_x_algo=self.mrx_algorithm)
            if self.game.get_game_status() != GameStatus.ONGOING:
                verbose_print("Game over", self.game_controller.verbose)
                self.game_controller.playing = False
                self.time_of_end = time.time()

        if self.time_of_end and time.time() - self.time_of_end > 2:
            self.game.reset()
            self.game_controller.playing = True
            self.time_of_end = None

    def render(self, display):
        self.game_visual.redraw()
        pygame.time.delay(150)

    def scene_cleanup(self):
        self.game_controller.playing = False
        self.game.quit()
