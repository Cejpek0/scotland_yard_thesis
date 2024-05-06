"""
File description: Define Scene class, which is base class for all scenes in game.

Author: Michal Cejpek (xcejpe05@stud.fit.vutbr.cz)
"""
from src.GameController import GameController
from src.GuiController import GuiController


class Scene:
    def __init__(self, game_controller: GameController | None, gui_controller: GuiController):
        self.game_controller = game_controller
        self.gui_controller = gui_controller
        self.prev_scene = None

    def update(self, user_input):
        """
        Update scene logic.
        :param user_input: User input.
        :return: None
        """
        pass

    def render(self, surface):
        """
        Render the scene.
        :param surface: Surface to render on.
        :return: None
        """
        pass

    def enter_scene(self):
        if len(self.game_controller.scene_stack) > 1:
            self.prev_scene = self.game_controller.scene_stack.top()
        self.game_controller.scene_stack.push(self)

    def scene_cleanup(self):
        """
        This method is called when a scene is being removed from the scene stack.
        :return:
        """
        pass

    def exit_scene(self):
        self.scene_cleanup()
