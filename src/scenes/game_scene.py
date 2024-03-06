import random
import time

import pygame
import ray
from src.GameController import GameController
from src.GuiController import GuiController
from src.Player import Player
from src.colors import *
import src.scotland_yard_game as scotland_yard_game

from src.scenes.scene import Scene


# Constants

class ScotlandYardScene(Scene):
    def __init__(self, game_controller: GameController, gui_controller: GuiController):
        Scene.__init__(self, game_controller, gui_controller)

        self.game = scotland_yard_game.ScotlandYardGame(training=False)
        self.cell_size = gui_controller.width // scotland_yard_game.GRID_SIZE

    def update(self, delta_time, actions):
        print("update")
        from src.GameController import UserActions
        if actions[UserActions.space.name]:
            from src.scenes.pause_menu import PauseMenu
            new_state = PauseMenu(self.game_controller, self.gui_controller)
            new_state.enter_scene()
        else:
            self.game.play_turn()
            if self.game.get_game_status() != scotland_yard_game.GameStatus.ONGOING:
                print("Game over")
                self.game_controller.playing = False
                time.sleep(3)
                self.exit_scene()

    def render(self, display):
        self.redraw()

    def create_players(self):
        self.game.players.clear()
        self.game.players.append(Player(0, color=RED, name="mr_x"))
        for i in range(self.game.number_of_cops):
            self.game.players.append(Player(i + 1, color=GREEN, name=f"cop_{i + 1}"))
        return self

    # -- BEGIN: CORE FUNCTIONS -- #

    def reset(self):
        # init game state
        self.game.turn_number = 0
        self.game.start_positions_mr_x.clear()
        self.game.start_positions_cops.clear()
        self.game.players.clear()
        # Create players
        self.game.create_players()

        self.game.start_positions_cops = self.game.generate_start_positions([], self.game.number_of_starting_positions_cops)
        self.game.start_positions_mr_x = self.game.generate_start_positions(self.game.start_positions_cops,
                                                                  self.game.number_of_starting_positions_mr_x)
        _start_positions_cops_temp = self.game.start_positions_cops.copy()
        for player in self.game.players:
            if player.number == 0:
                self.game.choose_start_position(player, random.choice(self.game.start_positions_mr_x))
            else:
                chosen_start_position = random.choice(_start_positions_cops_temp)
                self.game.choose_start_position(player, chosen_start_position)
                _start_positions_cops_temp.remove(chosen_start_position)

        if self.gui_controller.game_canvas is not None:
            self.to_draw_grid()
            self.redraw()
        return self

    def quit(self):
        ray.shutdown()
        pygame.quit()
        return self

    # -- END: CORE FUNCTIONS -- #

    # -- BEGIN: DRAW FUNCTIONS -- #

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
        (self.to_clear_grid()
         .to_draw_last_known_positions()
         .to_draw_players()
         .to_highlight_area_of_interest()
         )
        self.gui_controller.to_draw_text(text=f"Turn: {self.game.turn_number}", position=(10, 10))
        # set game status to display if game is over
        game_status = self.game.get_game_status()
        if game_status == scotland_yard_game.GameStatus.COPS_WON:
            self.gui_controller.to_draw_text(text="Cops won!", position=(10, 30))
        elif game_status == scotland_yard_game.GameStatus.MR_X_WON:
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
        return self

    def to_clear_grid(self):
        for row in range(self.game.grid_size):
            for col in range(self.game.grid_size):
                self.to_draw_rectangle_at_position((col, row), BLACK)
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

    def to_highlight_area_of_interest(self):
        if self.game.get_mr_x().last_known_position is not None:
            possible_mr_x_positions = self.game.get_circular_radius(
                self.game.get_mr_x().last_known_position, self.game.get_number_of_turns_since_last_reveal()
            )
        else:
            possible_mr_x_positions = []
            for starting_position in self.game.start_positions_mr_x:
                _possible_mr_x_positions = self.game.get_circular_radius(
                    starting_position,
                    self.game.get_number_of_turns_since_last_reveal()
                )
                for position in _possible_mr_x_positions:
                    if position not in possible_mr_x_positions:
                        possible_mr_x_positions.append(position)

        for position in possible_mr_x_positions:
            self.to_draw_rectangle_at_position(position, WHITE, 40)
        return self

    # -- END: DRAW FUNCTIONS -- #