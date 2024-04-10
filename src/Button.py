import pygame
from src.colors import *


# Button class
class Button:
    def __init__(self, text, x, y, width, height, screen, color, hover_color, text_color, text_hover_color,
                 border_color, border_color_hovered, border_size, border_size_hovered, value=None, font=None):
        self.value = value
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
        self.rect_border = pygame.Rect(x - border_size, y - border_size, width + border_size * 2,
                                       height + border_size * 2)
        self.rect_border_hovered = pygame.Rect(x - border_size_hovered, y - border_size_hovered,
                                               width + border_size_hovered * 2, height + border_size_hovered * 2)
        self.font = font
        if self.font is None:
            self.font = pygame.font.SysFont("Arial", 20)

        self.actual_color = self.color
        self.actual_text_color = self.text_color
        self.actual_border_color = self.border_color
        self.actual_border_size = self.border_size
        self.actual_rect_border = self.rect_border

    def draw(self):
        self.set_hover_style()
        pygame.draw.rect(self.screen, self.actual_border_color, self.actual_rect_border)
        pygame.draw.rect(self.screen, self.actual_color, self.rect)
        text_surface = self.font.render(self.text, True, self.actual_text_color)

        text_rect = text_surface.get_rect(center=self.rect.center)
        self.screen.blit(text_surface, text_rect)

    def set_hover_style(self):
        is_hovered = self.rect.collidepoint(pygame.mouse.get_pos())
        if is_hovered:
            self.actual_color = self.hover_color
            self.actual_text_color = self.text_hover_color
            self.actual_border_color = self.border_color_hovered
            self.actual_border_size = self.border_size_hovered
            self.actual_rect_border = self.rect_border_hovered
        else:
            self.actual_color = self.color
            self.actual_text_color = self.text_color
            self.actual_border_color = self.border_color
            self.actual_border_size = self.border_size
            self.actual_rect_border = self.rect_border

    def is_hovered(self):
        return self.rect.collidepoint(pygame.mouse.get_pos())
