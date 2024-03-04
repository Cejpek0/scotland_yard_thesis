from src.GameController import GameController
from src.GuiController import GuiController


class State:
    def __init__(self, game_controller: GameController, gui_controller: GuiController):
        self.game_controller = game_controller
        self.gui_controller = gui_controller
        self.prev_state = None

    def update(self, delta_time, actions):
        pass

    def render(self, surface):
        pass

    def enter_state(self):
        if len(self.game_controller.state_stack) > 1:
            self.prev_state = self.game_controller.state_stack[-1]
        self.game_controller.state_stack.append(self)

    def exit_state(self):
        self.game_controller.state_stack.pop()
