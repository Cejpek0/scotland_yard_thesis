import math
import random
import sys
import time
from enum import Enum

import gymnasium
import numpy as np
import pygame
import ray
from gymnasium import spaces
from ray.rllib import Policy
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune import register_env

from environments.rlib.FakeEnv import FakeEnv
from src.Player import Player
from src.colors import *

# Constants
WIDTH = 600
HEIGHT = 600
GRID_SIZE = 10
NUMBER_OF_STARTING_POSITIONS_COPS = 10
NUMBER_OF_STARTING_POSITIONS_MR_X = 5
MAX_NUMBER_OF_TURNS = 24
REVEAL_POSITION_TURNS = [3, 8, 13, 18, 24]
NAX_DISTANCE = math.ceil(math.sqrt(GRID_SIZE ** 2 + GRID_SIZE ** 2))

ALGORITHM_CHECKPOINT_DIR = "../tuned_results/"

MR_X_POLICY_SPEC = PolicySpec(
    observation_space=spaces.Box(
        low=np.array([
            0,  # current turn
            0,  # max turns
            0,  # next reveal
            0,  # position x
            0,  # position y
            0,  # position x of cop
            0,  # position y or cop
            -1,  # last known position x
            -1,  # last known position y
            0  # distance to cop
        ]),
        high=np.array([
            MAX_NUMBER_OF_TURNS,  # current turn
            MAX_NUMBER_OF_TURNS,  # max turns
            MAX_NUMBER_OF_TURNS,  # next reveal
            GRID_SIZE,  # position x
            GRID_SIZE,  # position y
            GRID_SIZE,  # position x of cop
            GRID_SIZE,  # position y or cop
            GRID_SIZE,  # last known position x
            GRID_SIZE,  # last known position y
            GRID_SIZE * 2  # distance to cop
        ]),
        dtype=np.float32
    ),
    action_space=spaces.Discrete(4),
)
COP_POLICY_SPEC = PolicySpec(
    observation_space=spaces.Box(
        low=np.array([
            0,  # current turn
            0,  # max turns
            0,  # next reveal
            0,  # position x
            0,  # position y
            -1,  # last known position x
            -1,  # last known position y
        ]),
        high=np.array([
            MAX_NUMBER_OF_TURNS,  # current turn
            MAX_NUMBER_OF_TURNS,  # max turns
            MAX_NUMBER_OF_TURNS,  # next reveal
            GRID_SIZE,  # position x
            GRID_SIZE,  # position y
            GRID_SIZE,  # last known position x
            GRID_SIZE,  # last known position y
        ]),
        dtype=np.float32
    ),
    action_space=spaces.Discrete(4),
)


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


class ScotlandYard:

    # -- BEGIN: CORE FUNCTIONS -- #
    def __init__(self, training: bool = False, number_of_cops: int = 1):
        self.number_of_cops = number_of_cops
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

        self.algorithm = None

        if not training:
            ray.init()

            my_config = PPOConfig()
            my_config["policies"] = {
                "mr_x_policy": MR_X_POLICY_SPEC,
                "cop_policy": COP_POLICY_SPEC,
            }
            my_config["policy_mapping_fn"] = \
                lambda agent_id, episode, worker, *kw: "mr_x_policy" if agent_id == "mr_x" else "cop_policy"

            def env_creator(env_config):
                return FakeEnv({})  # return an env instance

            register_env("scotland_env", env_creator)

            algo = PPO(env="scotland_env", config=my_config)
            algo.restore(
                "trained_policies")
            self.algorithm = algo

        # Create players
        self.create_players()

    def create_players(self):
        self.players.clear()
        self.players.append(Player(0, color=RED, name="mr_x"))
        for i in range(self.number_of_cops):
            self.players.append(Player(i + 1, color=GREEN, name=f"cop_{i + 1}"))
        return self

    def reset(self):
        # init game state
        self.turn_number = 1
        self.start_positions_mr_x.clear()
        self.start_positions_cops.clear()
        self.players.clear()
        # Create players
        self.create_players()

        self.start_positions_cops = self.generate_start_positions([], self.number_of_starting_positions_cops)
        self.start_positions_mr_x = self.generate_start_positions(self.start_positions_cops,
                                                                  self.number_of_starting_positions_mr_x)

        for player in self.players:
            if player.number == 0:
                self.choose_start_position(player, random.choice(self.start_positions_mr_x))
            else:
                self.choose_start_position(player, random.choice(self.start_positions_cops))

        if self.window is not None:
            self.to_draw_grid()
            self.redraw()
        return self

    def start_game(self):
        running = True
        started = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        started = True
            if started:
                self.play_turn()
                self.redraw()
                time.sleep(0.5)
                if self.get_game_status() != GameStatus.ONGOING:
                    running = False
                    time.sleep(3)
        return self

    def quit(self):
        ray.shutdown()
        pygame.quit()
        return self

    # -- END: CORE FUNCTIONS -- #

    # -- BEGIN: RL FUNCTIONS -- #
    def get_observations(self):
        # mr_x
        mr_x = self.get_mr_x()
        if mr_x.last_known_position is not None:
            obs_list_mrx = np.array([
                self.get_current_turn_number(),
                self.get_max_turns(),
                self.get_next_reveal_turn_number(),
                mr_x.position[0],
                mr_x.position[1],
                self.get_cops()[0].position[0],
                self.get_cops()[0].position[1],
                mr_x.last_known_position[0],
                mr_x.last_known_position[1],
                mr_x.get_distance_to(self.get_cops()[0])
            ]).astype(np.float32)
        else:
            obs_list_mrx = np.array([
                self.get_current_turn_number(),
                self.get_max_turns(),
                self.get_next_reveal_turn_number(),
                mr_x.position[0],
                mr_x.position[1],
                self.get_cops()[0].position[0],
                self.get_cops()[0].position[1],
                -1,
                -1,
                mr_x.get_distance_to(self.get_cops()[0])
            ]).astype(np.float32)

        # cops
        cop = self.get_cops()[0]

        if mr_x.last_known_position is not None:
            obs_list_cop = np.array([
                self.get_current_turn_number(),
                self.get_max_turns(),
                self.get_next_reveal_turn_number(),
                cop.position[0],
                cop.position[1],
                mr_x.last_known_position[0],
                mr_x.last_known_position[1],
            ]).astype(np.float32)
        else:
            obs_list_cop = np.array([
                self.get_current_turn_number(),
                self.get_max_turns(),
                self.get_next_reveal_turn_number(),
                cop.position[0],
                cop.position[1],
                -1,
                -1,
            ]).astype(np.float32)

        observations = {
            "mr_x": obs_list_mrx,
            "cop_1": obs_list_cop,
        }

        return observations

    def get_actions_for_players(self) -> dict:
        # Use the policy to obtain an action for the given player and observation
        observations = self.get_observations()

        action_is_valid = False
        direction = None
        mr_x = self.get_mr_x()
        cop = self.get_cops()[0]
        while not action_is_valid:
            generated_action = self.algorithm.compute_single_action(observations["mr_x"], policy_id="mr_x_policy")
            direction = Direction(generated_action)
            if self.is_valid_move(mr_x, direction):
                action_is_valid = True
        mr_x_direction = direction.value

        action_is_valid = False
        direction = None
        while not action_is_valid:
            generated_action = self.algorithm.compute_single_action(observations["cop_1"], policy_id="cop_policy")
            direction = Direction(generated_action)
            if self.is_valid_move(cop, direction):
                action_is_valid = True
        cop_1_direction = direction.value
        return {mr_x: mr_x_direction, cop: cop_1_direction}

    # -- END: RL FUNCTIONS -- #

    # --GAME CONTROL FUNCTIONS-- #

    # -- BEGIN: DRAW FUNCTIONS -- #
    def display(self):
        # Initialize Pygame
        pygame.init()
        self.window = pygame.display.set_mode((self.width, self.height), pygame.SRCALPHA)
        pygame.display.set_caption("Scotland yard AI")
        self.clock = pygame.time.Clock()
        self.reset()
        return self

    def set_draw_rectangle_at_position(self, position: (), color: (), alpha=255, small: bool = False):
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

        self.window.blit(rect_surface, (x_position, y_position))
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

    def to_draw_grid(self):
        self.window.fill(BLACK)

        # Draw horizontal lines
        for row in range(self.grid_size + 1):
            pygame.draw.line(self.window, WHITE, (0, row * self.cell_size), (self.width, row * self.cell_size), 1)
        # Draw vertical lines
        for col in range(self.grid_size + 1):
            pygame.draw.line(self.window, WHITE, (col * self.cell_size, 0), (col * self.cell_size, self.height), 1)
        return self

    def to_draw_players(self):
        for player in self.players:
            if player.position is not None:
                self.set_draw_rectangle_at_position(player.position, player.color, small=True)
        return self

    def to_clear_grid(self):
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.set_draw_rectangle_at_position((col, row), BLACK)
        return self

    def to_draw_last_known_positions(self):
        if self.get_mr_x().last_known_position is not None:
            self.set_draw_rectangle_at_position(self.get_mr_x().last_known_position, GRAY)
        return self

    def to_draw_text(self, text: str = "", color: () = WHITE, position: () = (0, 0)):
        font = pygame.font.SysFont("Arial", 20)
        text_surface = font.render(text, True, color)
        self.window.blit(text_surface, position)

        return self

    def to_highlight_start_positions(self):
        for position in self.start_positions_cops:
            self.set_draw_rectangle_at_position(position, GREEN, 128)
        for position in self.start_positions_mr_x:
            self.set_draw_rectangle_at_position(position, RED, 128)
        return self

    def redraw(self):
        self.to_clear_grid().to_draw_last_known_positions().to_draw_players().to_draw_text(
            text=f"Turn: {self.turn_number}", position=(10, 10))
        # set game status to display if game is over
        game_status = self.get_game_status()
        if game_status == GameStatus.COPS_WON:
            self.to_draw_text(text="Cops won!", position=(10, 30))
        elif game_status == GameStatus.MR_X_WON:
            self.to_draw_text(text="Mr X won!", position=(10, 30))
        pygame.display.flip()
        return self

    # -- END: DRAW FUNCTIONS -- #

    # --GAMEPLAY FUNCTIONS-- #

    def choose_start_position(self, player: Player, position: ()):
        player.set_start_position(position)
        return self

    def move_player(self, player: Player, direction: Direction):
        # print(f"Player {player.position} moves {direction}")
        if self.is_valid_move(player, direction):
            player.position = self.get_position_after_move(player, direction)
        else:
            print(f"P{player.number} tried to move {direction} from {player.position}")
            print(f"Would result in {self.get_position_after_move(player, direction)}")
            sys.stderr.write(f"Move {direction} is not valid\n")
            exit(1)
        return self

    def play_turn(self):
        print(f"Turn: {self.turn_number}")
        if self.turn_number in REVEAL_POSITION_TURNS:
            self.get_mr_x().mr_x_reveal_position()

        actions = {}
        all_actions_are_valid = False
        while not all_actions_are_valid:
            actions = self.get_actions_for_players()

            all_actions_are_valid = True
            for (player, action) in actions.items():
                direction = Direction(action)
                if not self.is_valid_move(player, direction):
                    all_actions_are_valid = False
                    break

        for (player, action) in actions.items():
            direction = Direction(action)
            self.move_player(player, direction)

        self.turn_number += 1
        self.redraw()

        return self

    # -- END: GAMEPLAY FUNCTIONS -- #

    # -- BEGIN: HELPER FUNCTIONS -- #

    def generate_start_positions(self, check_positions: [()], number_of_starting_positions: int) -> np.array:
        random_positions = []
        success = False
        # Generate random positions, cannot repeat
        while not success:
            random_position = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            # Check if random position is already in array
            if random_position not in random_positions and random_position not in check_positions:
                random_positions.append(random_position)
            if len(random_positions) == number_of_starting_positions:
                success = True
        return random_positions

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

    def is_valid_start_position(self, player: Player, position: ()):
        if player.number == 0 and position in self.start_positions_mr_x:
            return True
        elif player.number != 0 and position in self.start_positions_cops:
            return True
        return False

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

    def get_game_status(self) -> GameStatus:
        mr_x = self.get_mr_x()
        for cop in self.get_cops():
            if cop.position == mr_x.position:
                return GameStatus.COPS_WON
        if self.turn_number >= MAX_NUMBER_OF_TURNS:
            return GameStatus.MR_X_WON
        return GameStatus.ONGOING

    def get_player_by_name(self, name: str) -> Player | None:
        for player in self.players:
            if player.name == name:
                return player
        return None

    def log_start_info(self):
        print("Start positions cops: " + str(self.start_positions_cops))
        print("Start positions mr x: " + str(self.start_positions_mr_x))
        return self

    # -- END: HELPER FUNCTIONS -- #


if __name__ == '__main__':
    ScotlandYard().display().start_game().quit()
