import os, time, pygame
from enum import Enum

from src.GuiController import GuiController


# Enumerations
class UserActions(Enum):
    backspace = pygame.K_BACKSPACE
    enter = pygame.K_RETURN
    space = pygame.K_SPACE
    escape = pygame.K_ESCAPE
    mouse_left_down = pygame.MOUSEBUTTONDOWN
    mouse_right_down = pygame.MOUSEBUTTONDOWN
    mouse_left_up = pygame.MOUSEBUTTONUP
    mouse_right_up = pygame.MOUSEBUTTONUP
    mouse_moved = pygame.MOUSEMOTION


class GameController():
    def __init__(self):
        self.gui_controller = GuiController()
        self.assets_dir = os.path.join("assets")
        self.sprite_dir = os.path.join(self.assets_dir, "sprites")

        self.scene_stack = []
        from src.scenes.title import Title
        self.title_screen = Title(self, self.gui_controller)
        self.scene_stack.append(self.title_screen)
        pygame.init()

        self.running, self.playing = True, True
        self.user_actions = {}
        for action in UserActions:
            self.user_actions[action.name] = False
        self.dt, self.prev_time = 0, 0

        self.load_assets()

    def game_loop(self):
        while self.running:
            self.get_dt()
            self.get_events()
            self.update()
            self.render()
            self.reset_keys()

    def get_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.playing = False
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == UserActions.escape.value:
                    self.playing = False
                    self.running = False
                if event.key == UserActions.enter.value:  # Enter btn
                    self.user_actions[UserActions.enter.name] = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.user_actions[UserActions.mouse_left_down.name] = True
                if event.button == 3:
                    self.user_actions[UserActions.mouse_right_down.name] = True
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.user_actions[UserActions.mouse_left_up.name] = True
                if event.button == 3:
                    self.user_actions[UserActions.mouse_right_up.name] = True
            if event.type == pygame.MOUSEMOTION:
                self.user_actions[UserActions.mouse_moved.name] = True

    def update(self):
        self.scene_stack[-1].update(self.dt, self.user_actions)

    def render(self):
        self.scene_stack[-1].render(self.gui_controller.game_canvas)

        self.gui_controller.screen.blit(self.gui_controller.game_canvas, (0, 0))
        pygame.display.flip()

    def get_dt(self):
        now = time.time()
        self.dt = now - self.prev_time
        self.prev_time = now

    def load_assets(self):
        self.font = pygame.font.SysFont("Arial", 20)

    def reset_keys(self):
        for action in self.user_actions:
            self.user_actions[action] = False


if __name__ == "__main__":
    g = GameController()
    while g.running:
        g.game_loop()
