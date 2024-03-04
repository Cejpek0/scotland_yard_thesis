import os

import pygame

from src.GameController import GameController
from src.GuiController import GuiController
from src.states.state import State


class PauseMenu(State):
    def __init__(self, game_controller: GameController, gui_controller: GuiController):
        self.game_controller = game_controller
        self.gui_controller = gui_controller
        State.__init__(self, game_controller, gui_controller)
        # Set the menu
        self.menu_img = pygame.image.load(os.path.join(self.game_controller.assets_dir, "map", "menu.png"))
        self.menu_rect = self.menu_img.get_rect()
        self.menu_rect.center = (self.gui_controller.width * .85, self.gui_controller.height * .4)
        # Set the cursor and menu states
        self.menu_options = {1: "Exit"}
        self.index = 0
        self.cursor_img = pygame.image.load(os.path.join(self.game_controller.assets_dir, "map", "cursor.png"))
        self.cursor_rect = self.cursor_img.get_rect()
        self.cursor_pos_y = self.menu_rect.y + 38
        self.cursor_rect.x, self.cursor_rect.y = self.menu_rect.x + 10, self.cursor_pos_y

    def update(self, delta_time, actions):
        self.update_cursor(actions)
        if actions["action1"]:
            self.transition_state()
        if actions["action2"]:
            self.exit_state()

    def render(self, display):
        # render the game_controllerworld behind the menu, which is right before the pause menu on the stack
        # self.game_controller.state_stack[-2].render(display)
        self.prev_state.render(display)
        display.blit(self.menu_img, self.menu_rect)
        display.blit(self.cursor_img, self.cursor_rect)

    def transition_state(self):
        if self.menu_options[self.index] == "Exit":
            while len(self.game_controller.state_stack) > 1:
                self.game_controller.state_stack.pop()

    def update_cursor(self, actions):
        if actions['down']:
            self.index = (self.index + 1) % len(self.menu_options)
        elif actions['up']:
            self.index = (self.index - 1) % len(self.menu_options)
        self.cursor_rect.y = self.cursor_pos_y + (self.index * 32)
