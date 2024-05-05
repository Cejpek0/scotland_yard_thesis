"""
File description: This file contains the GameController class, which is the main class of the game. It is 
responsible for running the game loop, handling user input, and managing the game's scenes.

Author: Michal Cejpek(xcejpe05@stud.fit.vutbr.cz)

This code was inspired by the following source: https://github.com/ChristianD37/YoutubeTutorials/tree/master/Game%20States
Idea has been adopted, changed and expanded to fit the needs of the project.
"""

import pygame
import time
from enum import Enum

import ray

from TrainerDQN import TrainerDQN
from TrainerPPO import TrainerPPO
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
    p_key = pygame.K_p


class GameController():
    def __init__(self, verbose=False):
        print("GameController init")
        pygame.init()
        pygame.display.set_caption('Scotland Yard Thesis - xcejpe05')
        print("Pygame init")

        self.verbose = verbose
        self.font = None
        self.gui_controller = GuiController()
        self.load_assets()

        from src.scene_stack import SceneStack
        self.scene_stack = SceneStack()

        from src.scenes.title import Title
        self.title_screen = Title(self, self.gui_controller)
        self.scene_stack.push(self.title_screen)

        self.clock = pygame.time.Clock()
        self.running, self.playing = True, False
        self.user_actions = {}
        for action in UserActions:
            self.user_actions[action.name] = False
        self.dt, self.prev_time = 0, 0

        ray.init(num_gpus=0)

        trainer_ppo = TrainerPPO(playing=True)
        trainer_dqn = TrainerDQN(1, playing=True)
        self.algo_ppo = trainer_ppo.algo

        algo_dqn = trainer_dqn.algo
        algo_dqn.reset_config(trainer_dqn.play_config)
        self.algo_dqn = algo_dqn

    def game_loop(self):
        while self.running:
            self.get_events()
            if self.user_actions[UserActions.escape.name]:
                if len(self.scene_stack) > 1:
                    self.scene_stack.pop()
                else:
                    self.quit_game()
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
                    self.user_actions[UserActions.escape.name] = True
                if event.key == UserActions.enter.value:  # Enter btn
                    self.user_actions[UserActions.enter.name] = True
                if event.key == UserActions.space.value:  # Space btn
                    self.user_actions[UserActions.space.name] = True
                if event.key == UserActions.backspace.value:  # Backspace btn
                    self.user_actions[UserActions.backspace.name] = True
                if event.key == UserActions.p_key.value:  # P btn
                    self.user_actions[UserActions.p_key.name] = True
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
        self.scene_stack.top().update(self.dt, self.user_actions)

    def render(self):
        self.scene_stack.top().render(self.gui_controller.game_canvas)

        self.gui_controller.screen.blit(self.gui_controller.game_canvas, (0, 0))
        pygame.display.flip()

    def get_delta_time(self):
        current_time = time.time()
        self.dt = current_time - self.prev_time
        self.prev_time = current_time

    def quit_game(self):
        self.playing = False
        self.running = False
        self.scene_stack.top().scene_cleanup()
        pygame.quit()
        exit(0)

    def load_assets(self):
        self.font = pygame.font.SysFont("Arial", 20)

    def reset_keys(self):
        for action in self.user_actions:
            self.user_actions[action] = False
