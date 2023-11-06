import random
from enum import Enum
import numpy
import pygame

# Constants
WIDTH = 600
HEIGHT = 600
GRID_SIZE = 20
NUMBER_OF_STARTING_POSITIONS_AGENTS = 10
NUMBER_OF_STARTING_POSITIONS_MR_X = 5
NUMBER_OF_AGENTS = 3
MAX_NUMBER_OF_TURNS = 24
REVEAL_POSITION_TURNS = [3, 8, 13, 18, 24]

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
RED = (255, 0, 0)
GREEN = (0, 200, 0)


# Enumerations
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class GameStatus(Enum):
    AGENTS_WON = 1
    MR_X_WON = 2
    NOT_END = 3


class Player:
    def __init__(self, number: int, color: ()):
        self.last_known_position = None
        self.position = None
        self.number = number
        self.color = color
        self.start_position = None

    def set_start_position(self, position: ()):
        self.start_position = position
        self.position = position
        return self

    def move(self, direction: Direction):
        if direction == Direction.RIGHT:
            self.position = (self.position[0] + 1, self.position[1])
        elif direction == Direction.LEFT:
            self.position = (self.position[0] - 1, self.position[1])
        elif direction == Direction.UP:
            self.position = (self.position[0], self.position[1] - 1)
        elif direction == Direction.DOWN:
            self.position = (self.position[0], self.position[1] + 1)
        return self

    def get_distance_to(self, player) -> int:
        return abs(self.position[0] - player.position[0]) + abs(self.position[1] - player.position[1])

    def mr_x_reveal_position(self):
        if self.number == 0:
            self.last_known_position = self.position
        return self


class ScotlandYard:
    def __init__(self):
        self.turn_number = 1
        self.start_positions_mr_x = []
        self.start_positions_agents = []
        self.position_mr_x = None
        self.position_agents = []
        self.clock = None
        self.window = None
        self.width = WIDTH
        self.height = HEIGHT
        self.grid_size = GRID_SIZE
        self.cell_size = WIDTH // GRID_SIZE
        self.number_of_starting_positions_agents = NUMBER_OF_STARTING_POSITIONS_AGENTS
        self.number_of_starting_positions_mr_x = NUMBER_OF_STARTING_POSITIONS_MR_X
        self.players = []
        self.number_of_agents = NUMBER_OF_AGENTS
        # Create players
        self.players.append(Player(0, RED))
        for i in range(self.number_of_agents):
            self.players.append(Player(i + 1, GREEN))

    # --GAME CONTROL FUNCTIONS-- #

    def log_start_info(self):
        print("Start positions agents: " + str(self.start_positions_agents))
        print("Start positions mr x: " + str(self.start_positions_mr_x))
        return self

    def display(self):
        # Initialize Pygame
        pygame.init()
        self.window = pygame.display.set_mode((self.width, self.height), pygame.SRCALPHA)
        pygame.display.set_caption("Scotland yard AI")
        self.clock = pygame.time.Clock()
        self.reset()
        return self

    def draw_grid(self):
        self.window.fill(BLACK)

        # Draw horizontal lines
        for row in range(self.grid_size + 1):
            pygame.draw.line(self.window, WHITE, (0, row * self.cell_size), (self.width, row * self.cell_size), 1)
        # Draw vertical lines
        for col in range(self.grid_size + 1):
            pygame.draw.line(self.window, WHITE, (col * self.cell_size, 0), (col * self.cell_size, self.height), 1)
        return self

    def reset(self):
        # init game state
        self.turn_number = 1
        self.start_positions_mr_x = []
        self.start_positions_agents = []
        self.position_mr_x = None
        self.position_agents = []
        self.players = []
        self.number_of_agents = NUMBER_OF_AGENTS
        # Create players
        self.players.append(Player(0, RED))
        for i in range(self.number_of_agents):
            self.players.append(Player(i + 1, GREEN))

        self.draw_grid()
        pygame.display.flip()
        return self

    def start_game(self):
        running = True
        self.start_positions_agents = self.generate_start_positions([], self.number_of_starting_positions_agents)
        self.start_positions_mr_x = self.generate_start_positions(self.start_positions_agents,
                                                                  self.number_of_starting_positions_mr_x)
        self.log_start_info()
        self.highlight_start_positions()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self.quit()
        return self

    def quit(self):
        pygame.quit()
        return self

    def generate_start_positions(self, check_positions: [()], number_of_starting_positions: int) -> numpy.array:
        random_positions = []
        success = False
        # Generate random positions, cannot repeat
        while not success:
            random_position = (random.randint(0, self.grid_size), random.randint(0, self.grid_size))
            # Check if random position is already in array
            if random_position not in random_positions and random_position not in check_positions:
                random_positions.append(random_position)
            if len(random_positions) == number_of_starting_positions:
                success = True
        return random_positions

    def draw_rectangle_at_position(self, position: (), color: (), alpha=255):
        rect_surface = pygame.Surface((self.cell_size - 1, self.cell_size - 1),
                                      pygame.SRCALPHA)  # Create a surface with per-pixel alpha
        rect_surface.fill((color[0], color[1], color[2], alpha))
        self.window.blit(rect_surface, (position[0] * self.cell_size + 1, position[1] * self.cell_size + 1))
        pygame.display.flip()
        return self

    def highlight_start_positions(self):
        for position in self.start_positions_agents:
            self.draw_rectangle_at_position(position, GREEN, 128)
        for position in self.start_positions_mr_x:
            self.draw_rectangle_at_position(position, RED, 128)
        pygame.display.flip()
        return self

    def get_mr_x(self):
        return self.players[0]

    def get_agents(self):
        return self.players[1:]

    # --GAMEPLAY FUNCTIONS-- #
    def choose_start_position(self, player: Player, position: ()):
        player.set_start_position(position)
        if player.number == 0:
            self.start_positions_mr_x.remove(position)
        else:
            self.start_positions_agents.remove(position)
        return self

    def is_valid_start_position(self, player: Player, position: ()):
        if player.number == 0 and position in self.start_positions_mr_x:
            return True
        elif player.number != 0 and position in self.start_positions_agents:
            return True
        return False

    def move(self, player: Player, direction: Direction):
        return self

    def is_valid_move(self, player: Player, position: ()) -> bool:
        # Player can only move to position on grid
        if position[0] < 0 or position[0] > self.grid_size or position[1] < 0 or position[1] > self.grid_size:
            return False

        # Agent can only move to empty position or mr x position
        if player.number != 0:
            for player in self.players[1:]:
                if player.position == position:
                    return False
        return True

    def play_turn(self):
        return self

    def is_game_over(self) -> GameStatus:
        for agent in self.get_agents():
            if agent.position == self.get_mr_x().position:
                return GameStatus.AGENTS_WON
        if self.turn_number == MAX_NUMBER_OF_TURNS:
            return GameStatus.MR_X_WON
        return GameStatus.NOT_END


if __name__ == '__main__':
    ScotlandYard().display().start_game().quit()
