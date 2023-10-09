import pygame
from enum import Enum

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

BLOCK_SIZE = 20


class ScotlandYardGame:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = hasd
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('ScotlandYard')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        pass