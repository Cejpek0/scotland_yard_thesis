from src.GameController import GameController
from src.GuiController import GuiController


class Scene:
    def __init__(self, game_controller: GameController | None, gui_controller: GuiController):
        self.game_controller = game_controller
        self.gui_controller = gui_controller
        self.prev_scene = None

    def update(self, delta_time, actions):
        pass

    def render(self, surface):
        pass

    def enter_scene(self):
        if len(self.game_controller.scene_stack) > 1:
            self.prev_scene = self.game_controller.scene_stack.top()
        self.game_controller.scene_stack.push(self)

    def scene_cleanup(self):
        pass

    def exit_scene(self):
        self.scene_cleanup()
