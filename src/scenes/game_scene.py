import time

import pygame.time

from src.GameController import GameController
from src.GuiController import GuiController
from src.game.scotland_yard_game_logic import ScotlandYardGameLogic, GameStatus, DefinedAlgorithms
from src.game.scotland_yard_game_visual import ScotlandYardGameVisual, GRID_SIZE
from src.scenes.scene import Scene


# Constants

class ScotlandYardScene(Scene):
    def __init__(self, game_controller: GameController, gui_controller: GuiController, cop_algorithm=DefinedAlgorithms.PPO,
                 mrx_algorithm=DefinedAlgorithms.PPO):
        Scene.__init__(self, game_controller, gui_controller)

        self.time_of_end = None
        self.game = ScotlandYardGameLogic(False, cop_algorithm, mrx_algorithm)
        self.game_visual = ScotlandYardGameVisual(self.game, gui_controller)
        self.cell_size = gui_controller.width // GRID_SIZE

    def update(self, delta_time, actions):
        from src.GameController import UserActions

        if actions[UserActions.p_key.name]:
            from src.scenes.pause_menu import PauseMenu
            new_state = PauseMenu(self.game_controller, self.gui_controller)
            new_state.enter_scene()
        elif actions[UserActions.space.name] and self.game.get_game_status() == GameStatus.ONGOING:
            self.game_controller.playing = not self.game_controller.playing
        elif self.game_controller.playing:
            self.game.play_turn()
            if self.game.get_game_status() != GameStatus.ONGOING:
                print("Game overrrr")
                self.game_controller.playing = False
                self.time_of_end = time.time()

        if self.time_of_end and time.time() - self.time_of_end > 2:
            self.game.reset()
            self.game_controller.playing = True
            self.time_of_end = None

    def render(self, display):
        self.game_visual.redraw()   
        pygame.time.delay(100)
        

    def scene_cleanup(self):
        self.game_controller.playing = False
        self.game.quit()
