import time

from src.GameController import GameController
from src.GuiController import GuiController
from src.game.scotland_yard_game_logic import ScotlandYardGameLogic, GameStatus
from src.game.scotland_yard_game_visual import ScotlandYardGameVisual, GRID_SIZE
from src.scenes.scene import Scene


# Constants

class ScotlandYardScene(Scene):
    def __init__(self, game_controller: GameController, gui_controller: GuiController):
        Scene.__init__(self, game_controller, gui_controller)

        self.game = ScotlandYardGameLogic(training=False)
        self.game_visual = ScotlandYardGameVisual(self.game, gui_controller)
        self.cell_size = gui_controller.width // GRID_SIZE

    def update(self, delta_time, actions):
        print("update")
        from src.GameController import UserActions
        if actions[UserActions.space.name]:
            from src.scenes.pause_menu import PauseMenu
            new_state = PauseMenu(self.game_controller, self.gui_controller)
            new_state.enter_scene()
        else:
            self.game.play_turn()
            if self.game.get_game_status() != GameStatus.ONGOING:
                print("Game over")
                self.game_controller.playing = False
                time.sleep(3)
                self.exit_scene()

    def render(self, display):
        self.game_visual.redraw()
