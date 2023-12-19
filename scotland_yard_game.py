import random
import sys
from enum import Enum
import numpy as np
import pygame
from ray.rllib import Policy

# Constants
WIDTH = 600
HEIGHT = 600
GRID_SIZE = 20
NUMBER_OF_STARTING_POSITIONS_COPS = 10
NUMBER_OF_STARTING_POSITIONS_MR_X = 5
NUMBER_OF_COPS = 3
MAX_NUMBER_OF_TURNS = 24
REVEAL_POSITION_TURNS = [3, 8, 13, 18, 24]

SAVED_POLICY_DIR_MR_X = "trained_policies/policies/mr_x_policy"
SAVED_POLICY_DIR_COP = "trained_policies/policies/cop_policy"

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
RED = (255, 0, 0)
GREEN = (0, 200, 0)


# Enumerations
class Direction(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3


class GameStatus(Enum):
    COPS_WON = 1
    MR_X_WON = 2
    ONGOING = 3


class Player:
    def __init__(self, number: int, color: (), policy: Policy = None):
        self.last_known_position = None
        self.position = None
        self.number = number
        self.color = color
        self.start_position = None
        self.policy = policy

    def set_start_position(self, position: ()):
        self.start_position = position
        self.position = position
        return self

    def get_distance_to(self, *args) -> int:
        if len(args) == 1 and isinstance(args[0], Player):
            return self.get_distance_to(args[0].position)
        else:
            return abs(self.position[0] - args[0][0]) + abs(self.position[1] - args[0][1])

    def mr_x_reveal_position(self):
        if self.number == 0:
            self.last_known_position = self.position
        return self


class ScotlandYard:
    def __init__(self):
        self.turn_number = 1
        self.start_positions_mr_x = []
        self.start_positions_cops = []
        self.clock = None
        self.window = None
        self.width = WIDTH
        self.height = HEIGHT
        self.grid_size = GRID_SIZE
        self.cell_size = WIDTH // GRID_SIZE
        self.number_of_starting_positions_cops = NUMBER_OF_STARTING_POSITIONS_COPS
        self.number_of_starting_positions_mr_x = NUMBER_OF_STARTING_POSITIONS_MR_X
        self.players = []
        self.number_of_cops = NUMBER_OF_COPS
        
        self.mr_x_policy = None
        self.cop_policy = None

        self.mr_x_policy = PPOTrainer(config=your_mr_x_config)
        self.cop_policy = PPOTrainer(config=your_cop_config)

        self.mr_x_policy = ppo.PPOTrainer(config=your_mr_x_config, env="scotland_env")
        self.cop_policy = ppo.PPOTrainer(config=your_cop_config, env="scotland_env")

        self.mr_x_policy.restore(SAVED_POLICY_DIR_MR_X)
        self.cop_policy.restore(SAVED_POLICY_DIR_COP)
        
        # Create players
        self.players.append(Player(0, RED, mr_x_policy))
        for i in range(self.number_of_cops):
            self.players.append(Player(i + 1, GREEN, cop_policy))

    # --GAME CONTROL FUNCTIONS-- #

    def log_start_info(self):
        print("Start positions cops: " + str(self.start_positions_cops))
        print("Start positions mr x: " + str(self.start_positions_mr_x))
        return self

    def text_display(self):
        grid = np.full((self.grid_size, self.grid_size), " ")
        for player in self.players:
            if player.position is not None:
                if player.number == 0:
                    grid[player.position[1]][player.position[0]] = "X"
                else:
                    grid[player.position[1]][player.position[0]] = str(player.number)
        print(f"{grid}\n")

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
        self.start_positions_mr_x.clear()
        self.start_positions_cops.clear()
        self.players.clear()
        self.number_of_cops = NUMBER_OF_COPS
        # Create players
        self.players.append(Player(0, RED))
        for i in range(self.number_of_cops):
            self.players.append(Player(i + 1, GREEN))

        if self.window is not None:
            self.draw_grid()
            pygame.display.flip()
        return self

    def start_game(self):
        running = True
        self.start_positions_cops = self.generate_start_positions([], self.number_of_starting_positions_cops)
        self.start_positions_mr_x = self.generate_start_positions(self.start_positions_cops,
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

    def generate_start_positions(self, check_positions: [()], number_of_starting_positions: int) -> np.array:
        random_positions = []
        success = False
        # Generate random positions, cannot repeat
        while not success:
            random_position = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
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
        for position in self.start_positions_cops:
            self.draw_rectangle_at_position(position, GREEN, 128)
        for position in self.start_positions_mr_x:
            self.draw_rectangle_at_position(position, RED, 128)
        pygame.display.flip()
        return self

    def get_mr_x(self):
        return self.players[0]

    def get_cops(self):
        return self.players[1:]

    def get_cop_by_number(self, number: int) -> Player | None:
        for cop in self.get_cops():
            if cop.number == number:
                return cop
        return None

    def get_current_turn_number(self):
        return self.turn_number

    def get_max_turns(self):
        return MAX_NUMBER_OF_TURNS

    def get_next_reveal_turn_number(self):
        for turn_number in REVEAL_POSITION_TURNS:
            if turn_number > self.turn_number:
                return turn_number
        return MAX_NUMBER_OF_TURNS

    # --GAMEPLAY FUNCTIONS-- #
    def choose_start_position(self, player: Player, position: ()):
        player.set_start_position(position)
        if player.number == 0:
            self.start_positions_mr_x.remove(position)
        else:
            self.start_positions_cops.remove(position)
        return self

    def is_valid_start_position(self, player: Player, position: ()):
        if player.number == 0 and position in self.start_positions_mr_x:
            return True
        elif player.number != 0 and position in self.start_positions_cops:
            return True
        return False

    def move(self, player: Player, direction: Direction):
        return self

    def get_position_after_move(self, player: Player, direction: Direction) -> ():
        if direction == Direction.RIGHT:
            return player.position[0] + 1, player.position[1]
        elif direction == Direction.LEFT:
            return player.position[0] - 1, player.position[1]
        elif direction == Direction.UP:
            return player.position[0], player.position[1] - 1
        elif direction == Direction.DOWN:
            return player.position[0], player.position[1] + 1
        sys.stderr.write(f"Direction {direction} is not valid\n")

    def is_valid_move(self, player: Player, direction: Direction) -> bool:
        # Player can only move to position on grid
        
        
        position = self.get_position_after_move(player, direction)
        result = position[0] < 0 or position[0] >= GRID_SIZE or position[1] < 0 or position[1] >= GRID_SIZE
        if position[0] < 0 or position[0] >= GRID_SIZE or position[1] < 0 or position[1] >= GRID_SIZE:
            return False

        # Cop can only move to empty position or mr x position
        if player.number != 0:
            for player in self.players[1:]:
                if player.position == position:
                    return False
        return True
    
    def get_players_valid_moves(self, player: Player) -> [()]:
        valid_moves = []
        for direction in Direction:
            if self.is_valid_move(player, direction):
                valid_moves.append(direction)
        return valid_moves
    
    def get_players_valid_moves_mask(self, player: Player) -> [int]:
        valid_moves = []
        for direction in Direction:
            if self.is_valid_move(player, direction):
                valid_moves.append(1)
            else:
                valid_moves.append(0)
        print(valid_moves)
        return valid_moves

    def move_player(self, player: Player, direction: Direction):
        #print(f"Player {player.position} moves {direction}")
        if self.is_valid_move(player, direction):
            player.position = self.get_position_after_move(player, direction)
        else:
            print(f"P{player.number} tried to move {direction} from {player.position}")
            print(f"Would result in {self.get_position_after_move(player, direction)}")
            print(self.get_players_valid_moves_mask(player))
            print(self.is_valid_move(player, direction))
            sys.stderr.write(f"Move {direction} is not valid\n")
            exit(1)
        return self

    def play_turn(self):
        return self

    def is_game_over(self) -> GameStatus:
        mr_x = self.get_mr_x()
        for cop in self.get_cops():
            if cop.position == mr_x.position:
                return GameStatus.COPS_WON
        if self.turn_number >= MAX_NUMBER_OF_TURNS:
            return GameStatus.MR_X_WON
        return GameStatus.ONGOING


if __name__ == '__main__':
    ScotlandYard().display().start_game().quit()
