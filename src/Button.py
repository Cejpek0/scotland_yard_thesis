import pygame
from src.colors import *


# Button class
class Button:
    def __init__(self, text, x, y, width, height, screen, color, hover_color, text_color,
                 text_hover_color, border_color, border_color_hovered, border_size, border_size_hovered, font=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.text = text
        self.text_color = text_color
        self.text_hover_color = text_hover_color
        self.border_color = border_color
        self.border_color_hovered = border_color_hovered
        self.border_size = border_size
        self.border_size_hovered = border_size_hovered
        self.screen = screen
        self.font = font
        if self.font is None:
            self.font = pygame.font.SysFont("Arial", 20)

    def draw(self):
        pygame.draw.rect(self.screen, self.border_color, self.rect, self.border_size)
        pygame.draw.rect(self.screen, self.color, self.rect)
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        self.screen.blit(text_surface, text_rect)

    def is_hovered(self):
        return self.rect.collidepoint(pygame.mouse.get_pos())
