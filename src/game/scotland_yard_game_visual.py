"""
File description: Visual representation of the Scotland Yard game.

Author: Michal Cejpek (xcejpe05@stud.fit.vutbr.cz)
"""
import pygame

from src.GuiController import GuiController
from src.Player import Player
from src.colors import *
from src.game.scotland_yard_game_logic import ScotlandYardGameLogic, GameStatus, GRID_SIZE


# Constants

class ScotlandYardGameVisual:
    def __init__(self, game_instance: ScotlandYardGameLogic, gui_controller: GuiController):
        self.grid_size = GRID_SIZE
        self.game = game_instance
        self.gui_controller = gui_controller
        self.cell_size = gui_controller.width // self.grid_size

    def to_draw_rectangle_at_position(self, position: (), color: (), alpha=255, small: bool = False):
        if small:
            rect_width = self.cell_size // 2 - 1
            rect_height = self.cell_size // 2 - 1
        else:
            rect_width = self.cell_size - 2
            rect_height = self.cell_size - 2

        rect_surface = pygame.Surface((rect_width, rect_height), pygame.SRCALPHA)
        rect_surface.fill((color[0], color[1], color[2], alpha))

        x_position = position[0] * self.cell_size + (self.cell_size - rect_width) // 2 if small else position[
                                                                                                         0] * self.cell_size + 1
        y_position = position[1] * self.cell_size + (self.cell_size - rect_height) // 2 if small else position[
                                                                                                          1] * self.cell_size + 1

        self.gui_controller.game_canvas.blit(rect_surface, (x_position, y_position))
        return self

    def redraw(self):
        """
        Redraws the whole game canvas based on the current game state.
        :return: self
        """
        (self.to_clear_grid()
         .to_draw_last_known_positions()
         .to_draw_players()
         .to_highlight_area_of_interest()
         )
        self.gui_controller.to_draw_text(text=f"Turn: {self.game.round_number}", position=(10, 10))
        # set game status to display if game is over
        game_status = self.game.get_game_status()
        if game_status == GameStatus.COPS_WON:
            self.gui_controller.to_draw_text(text="Cops won!", position=(10, 30))
        elif game_status == GameStatus.MR_X_WON:
            self.gui_controller.to_draw_text(text="Mr X won!", position=(10, 30))
        pygame.display.flip()
        return self

    def to_draw_grid(self):
        self.gui_controller.game_canvas.fill(BLACK)

        # Draw horizontal lines
        for row in range(self.game.grid_size + 1):
            pygame.draw.line(self.gui_controller.game_canvas, WHITE, (0, row * self.cell_size),
                             (self.gui_controller.width, row * self.cell_size), 1)
        # Draw vertical lines
        for col in range(self.game.grid_size + 1):
            pygame.draw.line(self.gui_controller.game_canvas, WHITE, (col * self.cell_size, 0),
                             (col * self.cell_size, self.gui_controller.height), 1)
        return self

    def to_draw_players(self):
        for player in self.game.players:
            if player.position is not None:
                self.to_draw_rectangle_at_position(player.position, player.color, small=True)
                self.draw_player_number(player)
        return self

    def to_clear_grid(self):
        for row in range(self.game.grid_size):
            for col in range(self.game.grid_size):
                self.to_draw_rectangle_at_position((col, row), BLACK)
        self.to_draw_grid()
        return self

    def to_draw_last_known_positions(self):
        if self.game.get_mr_x().last_known_position is not None:
            self.to_draw_rectangle_at_position(self.game.get_mr_x().last_known_position, GRAY)
        return self

    def to_highlight_start_positions(self):
        for position in self.game.start_positions_cops:
            self.to_draw_rectangle_at_position(position, GREEN, 128)
        for position in self.game.start_positions_mr_x:
            self.to_draw_rectangle_at_position(position, RED, 128)
        return self

    def draw_player_number(self, player: Player):
        self.gui_controller.to_draw_text(text=f"{player.number}", position=(
            player.position[0] * self.cell_size + 5, player.position[1] * self.cell_size + 5))
        return self

    def to_highlight_area_of_interest(self):
        possible_mr_x_positions = self.game.get_possible_mr_x_positions()

        for position in possible_mr_x_positions:
            self.to_draw_rectangle_at_position(position, WHITE, 40)
        return self
