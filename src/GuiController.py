"""
File description: This file contains the GuiController class, which is responsible for managing the game's GUI.

Author: Michal Cejpek (xcejpe05@stud.fit.vutbr.cz)
"""
import pygame

from src.colors import *
WIDTH = 600
HEIGHT = 600
CELL_SIZE = 15


class GuiController:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.game_canvas = pygame.Surface((self.width, self.height))
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.SRCALPHA)

    def set_caption(self, caption: str):
        pygame.display.set_caption(caption)
        return self

    def to_draw_text(self, text: str = "", color: () = WHITE, position: () = (0, 0)):
        font = pygame.font.SysFont("Arial", 20)
        text_surface = font.render(text, True, color)
        self.game_canvas.blit(text_surface, position)

        return self
