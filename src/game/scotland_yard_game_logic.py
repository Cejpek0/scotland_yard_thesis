import math
import os
import random
import sys
from enum import Enum

import numpy as np
import ray
from gymnasium import spaces

import torch
from jinja2.nodes import Name
from ray.rllib.algorithms import PPOConfig, PPO, DQNConfig, DQN
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

from src.Cop import Cop
from src.MrX import MrX
from src.Player import Player
from src.colors import *
from src.environments.rlib.FakeEnv import FakeEnv
from src.helper import verbose_print

# Constants

GRID_SIZE = 15
MAX_DISTANCE = round(math.sqrt(GRID_SIZE ** 2 + GRID_SIZE ** 2), 2)
NUMBER_OF_STARTING_POSITIONS_COPS = 10
NUMBER_OF_STARTING_POSITIONS_MR_X = 5
MAX_NUMBER_OF_TURNS = 24
REVEAL_POSITION_ROUNDS = [3, 8, 13, 18, 24]

ALGORITHM_CHECKPOINT_DIR = "../tuned_results/"

mrx_observation_space = spaces.Box(low=np.array([
    0,  # current turn
    0,  # max turns
    0,  # next reveal
    0,  # number of inactivity rounds
    0,  # position x
    0,  # position y
    -1,  # last known position x
    -1,  # last known position y
    -1,  # distance to last known position
    0,  # position x of cop_1
    0,  # position y or cop_1
    0,  # distance to cop_1
    0,  # position x of cop_2
    0,  # position y or cop_2
    0,  # distance to cop_2
    0,  # position x of cop_3
    0,  # position y or cop_3
    0,  # distance to cop_3
], dtype=np.float32), high=np.array([
    MAX_NUMBER_OF_TURNS,  # current turn
    MAX_NUMBER_OF_TURNS,  # max turns
    MAX_NUMBER_OF_TURNS,  # next reveal
    MAX_NUMBER_OF_TURNS,  # number of inactivity rounds
    GRID_SIZE,  # position x
    GRID_SIZE,  # position y
    GRID_SIZE,  # last known position x
    GRID_SIZE,  # last known position y
    GRID_SIZE * 2,  # distance to last known position
    GRID_SIZE,  # position x of cop_1
    GRID_SIZE,  # position y or cop_1
    GRID_SIZE * 2,  # distance to cop_1
    GRID_SIZE,  # position x of cop_2
    GRID_SIZE,  # position y or cop_2
    GRID_SIZE * 2,  # distance to cop_2
    GRID_SIZE,  # position x of cop_3
    GRID_SIZE,  # position y or cop_3
    GRID_SIZE * 2,  # distance to cop_3
]), dtype=np.float32)

general_cop_observation_space = spaces.Box(low=np.array([
    0,  # current turn
    0,  # max turns
    0,  # next reveal
    0,  # number of inactivity rounds
    0,  # position x
    0,  # position y
    -1,  # last known position x of mr x
    -1,  # last known position y of mr x
    -1,  # distance to last known position of mr x
    0,  # position x of other cop_1
    0,  # position y or other cop_1
    0,  # distance to other cop_1
    0,  # position x of other cop_2
    0,  # position y or other cop_2
    0,  # distance to other cop_2
    0,  # position x of closest point of interest
    0,  # position y of closest point of interest
    0,  # distance to the closest point of interest
    0,  # inside or outside area of interest
], dtype=np.float32), high=np.array([
    MAX_NUMBER_OF_TURNS,  # current turn
    MAX_NUMBER_OF_TURNS,  # max turns
    MAX_NUMBER_OF_TURNS,  # next reveal
    MAX_NUMBER_OF_TURNS,  # number of inactivity rounds
    GRID_SIZE,  # position x
    GRID_SIZE,  # position y
    GRID_SIZE,  # last known position x of mr x
    GRID_SIZE,  # last known position y of mr x
    MAX_DISTANCE,  # distance to last known position of mr x
    GRID_SIZE,  # position x of other cop_1
    GRID_SIZE,  # position y or other cop_1
    MAX_DISTANCE,  # distance to other cop_1
    GRID_SIZE,  # position x of other cop_2
    GRID_SIZE,  # position y or other cop_2
    MAX_DISTANCE,  # distance to other cop_2,
    GRID_SIZE,  # position x of closest point of interest
    GRID_SIZE,  # position y of closest point of interest
    MAX_DISTANCE,  # distance to the closest point of interest
    1,  # inside or outside area of interest
]), dtype=np.float32)

MR_X_POLICY_SPEC = PolicySpec(
    observation_space=mrx_observation_space,
    action_space=spaces.Discrete(9),
)
COP_POLICY_SPEC = PolicySpec(
    observation_space=general_cop_observation_space,
    action_space=spaces.Discrete(9),
)


# Enumerations
class Direction(Enum):
    STOP = 0
    UP = 1
    UP_RIGHT = 2
    RIGHT = 3
    DOWN_RIGHT = 4
    DOWN = 5
    DOWN_LEFT = 6
    LEFT = 7
    UP_LEFT = 8


class GameStatus(Enum):
    COPS_WON = 1
    MR_X_WON = 2
    ONGOING = 3


class DefinedAlgorithms(Enum):
    PPO = 1
    DQN = 2
    RANDOM = 3


class ScotlandYardGameLogic:
    def __init__(self, training=False, verbose=False, cop_algorithm=DefinedAlgorithms.PPO,
                 mrx_algorithm=DefinedAlgorithms.PPO):
        self.verbose = verbose
        self.cop_algorithm = cop_algorithm
        self.mrx_algorithm = mrx_algorithm

        self.number_of_cops = 3
        self.round_number = 0
        self.start_positions_mr_x = []
        self.start_positions_cops = []
        self.grid_size = GRID_SIZE
        self.number_of_starting_positions_cops = NUMBER_OF_STARTING_POSITIONS_COPS
        self.number_of_starting_positions_mr_x = NUMBER_OF_STARTING_POSITIONS_MR_X
        self.players = []
        self.playing_player_index = 0

        self.agents_previous_locations = {"mr_x": None, "cop_1": None, "cop_2": None, "cop_3": None}
        self.agents_is_in_previous_location_count = {"mr_x": 0, "cop_1": 0, "cop_2": 0, "cop_3": 0}

        if not training:
            ray.init()
            self.cop_model_ppo = None
            self.mrx_model_ppo = None

            self.policy_mrx_ppo = None
            self.policy_cop_ppo = None

            self.cop_model_dqn = None
            self.mrx_model_dqn = None

            def env_creator(env_config):
                return FakeEnv({})

            register_env("scotland_env", env_creator)

            def policy_mapping_fn(agent_id, episode, worker):
                return "mr_x_policy" if agent_id == "mr_x" else "cop_policy"

            ppo_config = PPOConfig()

            ppo_config["policies"] = {
                "mr_x_policy": MR_X_POLICY_SPEC,
                "cop_policy": COP_POLICY_SPEC,
            }

            ppo_config["policy_mapping_fn"] = policy_mapping_fn
            ppo_config.framework("torch")

            # Set the config object's env.
            algo_ppo = PPO(env="scotland_env", config=ppo_config)
            # check if trained policies exist
            directory = "trained_policies"
            assert os.path.exists(directory), f"Directory {directory} does not exist"
            algo_ppo.restore(directory)

            self.algo_ppo = algo_ppo

            dqn_config = (DQNConfig()
                          .training(
                lr=0.001,
                gamma=0.99,
                target_network_update_freq=10,
                double_q=True,
                dueling=True,
                num_atoms=1,
                noisy=True,
                n_step=3, )
                          .rollouts(observation_filter="MeanStdFilter"))


            replay_config = {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 50000,
                "prioritized_replay_alpha": 0.5,
                "prioritized_replay_beta": 0.5,
                "prioritized_replay_eps": 3e-6,
            }

            exploration_config = {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.05,
            }

            dqn_config["replay_config"] = replay_config
            dqn_config["exploration_config"] = exploration_config

            dqn_config.evaluation_config = {
                "evaluation_interval": 10,
                "evaluation_num_episodes": 10,
            }

            dqn_config["policies"] = {
                "mr_x_policy": MR_X_POLICY_SPEC,
                "cop_policy": COP_POLICY_SPEC,
            }

            dqn_config["policy_mapping_fn"] = policy_mapping_fn
            dqn_config.framework("torch")
            dqn_config["reuse_actors"] = True

            # Set the config object's env.
            algo_dqn = DQN(env="scotland_env", config=dqn_config)

            # check if trained policies exist
            directory = "trained_policies_dqn"

            assert os.path.exists(directory), f"Directory {directory} does not exist"
            algo_dqn.restore(directory)
            self.algo_dqn = algo_dqn

        verbose_print(f"Cop algorithm: {cop_algorithm}", self.verbose)
        verbose_print(f"Mr X algorithm: {mrx_algorithm}", self.verbose)

        # Create players
        self.reset()

    def create_players(self):
        self.players.clear()
        self.players.append(MrX(0, color=RED, name="mr_x"))
        for i in range(self.number_of_cops):
            self.players.append(Cop(i + 1, color=GREEN, name=f"cop_{i + 1}"))
        return self

    # -- BEGIN: CORE FUNCTIONS -- #

    def reset(self):
        # init game state
        self.round_number = 0
        self.playing_player_index = 0
        self.start_positions_mr_x.clear()
        self.start_positions_cops.clear()
        self.agents_is_in_previous_location_count = {"mr_x": 0, "cop_1": 0, "cop_2": 0, "cop_3": 0}
        self.players.clear()
        # Create players
        self.create_players()

        self.start_positions_cops = self.generate_start_positions([], self.number_of_starting_positions_cops)
        self.start_positions_mr_x = self.generate_start_positions(self.start_positions_cops,
                                                                  self.number_of_starting_positions_mr_x)
        _start_positions_cops_temp = self.start_positions_cops.copy()
        for player in self.players:
            if player.number == 0:
                self.choose_start_position(player, random.choice(self.start_positions_mr_x))
            else:
                chosen_start_position = random.choice(_start_positions_cops_temp)
                self.choose_start_position(player, chosen_start_position)
                _start_positions_cops_temp.remove(chosen_start_position)
        self.agents_previous_locations = dict(mr_x=self.get_mr_x().position,
                                              cop_1=self.get_player_by_number(1).position,
                                              cop_2=self.get_player_by_number(2).position,
                                              cop_3=self.get_player_by_number(3).position)

        return self

    def quit(self):
        return self

    # -- END: CORE FUNCTIONS -- #

    # -- BEGIN: RL FUNCTIONS -- #

    def get_square_radius(self, position: (int, int), r: int):
        positions = []
        for x in range(position[0] - r, position[0] + r + 1):
            for y in range(position[1] - r, position[1] + r + 1):
                if self.is_position_inside_grid((x, y)):
                    positions.append((x, y))
        return positions

    def get_closest_position(self, position: (int, int), positions: [(int, int)]):
        closest_position = None
        min_distance = MAX_DISTANCE
        for pos in positions:
            distance = self.get_distance_between_positions(position, pos)
            if distance < min_distance:
                min_distance = distance
                closest_position = pos
        return closest_position

    def get_observations(self):
        cop_1 = self.get_player_by_number(1)
        cop_2 = self.get_player_by_number(2)
        cop_3 = self.get_player_by_number(3)
        mr_x = self.get_mr_x()

        # mr_x
        if self.agents_is_in_previous_location_count["mr_x"] > MAX_NUMBER_OF_TURNS:
            verbose_print(f"ERROR:{self.agents_is_in_previous_location_count}", self.verbose)
        obs_list_mrx = np.array([
            self.get_current_round_number(),
            self.get_max_rounds(),
            self.get_next_reveal_round_number(),
            self.agents_is_in_previous_location_count["mr_x"],
            mr_x.position[0],
            mr_x.position[1],
            mr_x.last_known_position[0] if mr_x.last_known_position is not None else -1,
            mr_x.last_known_position[1] if mr_x.last_known_position is not None else -1,
            mr_x.get_distance_to(mr_x.last_known_position) if mr_x.last_known_position is not None else -1,
            cop_1.position[0],
            cop_1.position[1],
            mr_x.get_distance_to(cop_1.position),
            cop_2.position[0],
            cop_2.position[1],
            mr_x.get_distance_to(cop_2.position),
            cop_3.position[0],
            cop_3.position[1],
            mr_x.get_distance_to(cop_3.position),
        ]).astype(np.float32)

        # cops
        cops_observations = []

        possible_mr_x_positions = self.get_possible_mr_x_positions()

        for cop_number in range(1, self.number_of_cops + 1):
            cop = self.get_player_by_number(cop_number)
            if self.agents_is_in_previous_location_count[cop.name] > MAX_NUMBER_OF_TURNS:
                verbose_print(f"ERROR:{self.agents_is_in_previous_location_count}", self.verbose)
            obs_list_cop = np.array([
                self.get_current_round_number(),
                self.get_max_rounds(),
                self.get_next_reveal_round_number(),
                self.agents_is_in_previous_location_count[cop.name],
                cop.position[0],
                cop.position[1],
                mr_x.last_known_position[0] if mr_x.last_known_position is not None else -1,
                mr_x.last_known_position[1] if mr_x.last_known_position is not None else -1,
                cop.get_distance_to(mr_x.last_known_position) if mr_x.last_known_position is not None else -1,
            ]).astype(np.float32)

            for other_cop in self.get_cops():
                if other_cop.number == cop_number:
                    continue
                obs_list_cop = np.append(obs_list_cop, np.array([
                    other_cop.position[0],
                    other_cop.position[1],
                    cop.get_distance_to(other_cop.position)
                ]).astype(np.float32))

            closest_position = self.get_closest_position(
                cop.position,
                possible_mr_x_positions
            )
            distance_to_closest_position = cop.get_distance_to(closest_position)
            obs_list_cop = np.append(obs_list_cop, np.array([
                closest_position[0],
                closest_position[1],
                distance_to_closest_position,
                1 if cop.position in possible_mr_x_positions else 0
            ]).astype(np.float32))

            cops_observations.append(obs_list_cop)

        observations = {
            "mr_x": obs_list_mrx
        }
        for i in range(len(cops_observations)):
            observations[f"cop_{i + 1}"] = cops_observations[i]

        return observations

    def get_random_action(self):
        valid_actions = self.get_players_valid_moves(self.get_current_player())
        generated_action = random.choice(valid_actions)
        if self.is_valid_move(self.get_current_player(), Direction(generated_action)):
            return Direction(generated_action)
        else:
            raise Exception("Generated action is not valid")

    def get_action_for_player(self, player: Player) -> Direction:
        # Use the policy to obtain an action for the given player and observation
        observations = self.get_observations()

        count = 0
        direction = None
        action_is_valid = False
        observations_tensor = torch.tensor(observations[player.name], dtype=torch.float32)
        observations_tensor = torch.unsqueeze(observations_tensor, 0)
        while not action_is_valid:
            if count < 100:
                if player.is_mr_x():
                    if self.mrx_algorithm == DefinedAlgorithms.DQN:
                        generated_action = self.algo_dqn.compute_single_action(observations[player.name],
                                                                               policy_id="mr_x_policy")
                    else:
                        model_out, _ = self.policy_mrx_ppo.model({"obs": observations_tensor})
                        action_dist = self.policy_mrx_ppo.dist_class(model_out, self.policy_mrx_ppo.model)
                        generated_action = action_dist.sample().item()
                        generated_action = self.algo_ppo.compute_single_action(observations[player.name],
                                                                               policy_id="mr_x_policy")
                else:
                    if self.cop_algorithm == DefinedAlgorithms.DQN:
                        generated_action = self.algo_dqn.compute_single_action(observations[player.name],
                                                                               policy_id="cop_policy")
                    else:
                        model_out, _ = self.policy_cop_ppo.model({"obs": observations_tensor})
                        action_dist = self.policy_cop_ppo.dist_class(model_out, self.policy_cop_ppo.model)
                        generated_action = action_dist.sample().item()
                        generated_action = self.algo_ppo.compute_single_action(observations[player.name],
                                                                               policy_id="cop_policy")

            else:
                generated_action = self.get_random_action()
                verbose_print(f"Generated random action after 100 tries", self.verbose)
            direction = Direction(generated_action)
            if self.is_valid_move(player, direction):
                action_is_valid = True
            count += 1
        return direction

    # -- END: RL FUNCTIONS -- #

    # --GAME CONTROL FUNCTIONS-- #

    def choose_start_position(self, player: Player, position: ()):
        player.set_start_position(position)
        return self

    def move_player(self, player: Player, direction: Direction):
        # verbose_print(f"Player {player.position} moves {direction}", self.verbose)
        if self.is_valid_move(player, direction):
            player.position = self.get_position_after_move(player, direction)
        else:
            verbose_print(f"{player.name} tried to move {direction} from {player.position}", self.verbose)
            verbose_print(f"This move would result in {self.get_position_after_move(player, direction)}", self.verbose)
            sys.stderr.write(f"Move {direction} is not valid\n")
            exit(1)
        return self

    def get_current_player(self) -> Player:
        return self.players[self.playing_player_index]

    def play_turn(self, action: Direction | Name = None, verbose=False):
        if self.playing_player_index == 0:
            self.round_number += 1

        player = self.get_current_player()
        verbose_print(f"Player {player.name} is playing and player {'is' if player.is_mr_x() else 'is not'} mr x",
                      self.verbose)
        verbose_print(f"Generated action: {action}", self.verbose)
        verbose_print(f"Mrx_algo is {self.mrx_algorithm.name} and cop_algo is {self.cop_algorithm.name}", self.verbose)

        if action is None:
            if player.is_mr_x():
                verbose_print("Player is really mr_x", self.verbose)
                if self.mrx_algorithm is DefinedAlgorithms.RANDOM:
                    verbose_print("His algo is random", self.verbose)
                    action = self.get_random_action()
                else:
                    verbose_print("His algo is not random", self.verbose)
                    action = self.get_action_for_player(player)
            elif player.is_cop():
                verbose_print("Player is really cop", self.verbose)
                if self.cop_algorithm is DefinedAlgorithms.RANDOM:
                    verbose_print("His algo is random", self.verbose)
                    action = self.get_random_action()
                else:
                    verbose_print("His algo is not random", self.verbose)
                    action = self.get_action_for_player(player)

        self.move_player(player, action)

        if player.position == self.agents_previous_locations[player.name]:
            self.agents_is_in_previous_location_count[player.name] += 1
        else:
            self.agents_is_in_previous_location_count[player.name] = 0
        self.agents_previous_locations[player.name] = player.position

        if self.round_number in REVEAL_POSITION_ROUNDS and player.is_mr_x():
            self.get_mr_x().mr_x_reveal_position()
        self.playing_player_index = (self.playing_player_index + 1) % len(self.players)

        #verbose_print(self.get_rewards_fake(), self.verbose)

        return self

    # -- END: GAMEPLAY FUNCTIONS -- #

    # -- BEGIN: HELPER FUNCTIONS -- #

    def is_position_inside_grid(self, position: (int, int)) -> bool:
        if position[0] < 0 or position[0] >= self.grid_size or position[1] < 0 or position[1] >= self.grid_size:
            return False
        return True

    def get_number_of_rounds_since_last_reveal(self):
        return self.round_number - self.get_previous_reveal_round_number()

    def get_previous_reveal_round_number(self):
        previous_reveal_turn_number = 0
        for reveal_turn_number in REVEAL_POSITION_ROUNDS:
            if reveal_turn_number <= self.round_number:
                previous_reveal_turn_number = reveal_turn_number
        return previous_reveal_turn_number

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

    def get_mr_x(self) -> MrX:
        return self.players[0]

    def get_cops(self):
        return self.players[1:]

    def get_player_by_number(self, number: int) -> Player | None:
        for player in self.players:
            if player.number == number:
                return player
        return None

    def get_current_round_number(self):
        return self.round_number

    def get_max_rounds(self):
        return MAX_NUMBER_OF_TURNS

    def get_next_reveal_round_number(self):
        for round_number in REVEAL_POSITION_ROUNDS:
            if round_number > self.round_number:
                return round_number
        return MAX_NUMBER_OF_TURNS

    def is_valid_start_position(self, player: Player, position: ()):
        if position[0] < 0 or position[0] >= self.grid_size or position[1] < 0 or position[1] >= self.grid_size:
            return False
        if player.number == 0 and position in self.start_positions_mr_x:
            return True
        elif player.number != 0 and position in self.start_positions_cops:
            return True
        return False

    def get_position_after_move(self, player: Player, direction: Direction) -> ():
        if direction == Direction.STOP:
            return player.position
        elif direction == Direction.UP:
            return player.position[0], player.position[1] - 1
        elif direction == Direction.UP_RIGHT:
            return player.position[0] + 1, player.position[1] - 1
        elif direction == Direction.RIGHT:
            return player.position[0] + 1, player.position[1]
        elif direction == Direction.DOWN_RIGHT:
            return player.position[0] + 1, player.position[1] + 1
        elif direction == Direction.DOWN:
            return player.position[0], player.position[1] + 1
        elif direction == Direction.DOWN_LEFT:
            return player.position[0] - 1, player.position[1] + 1
        elif direction == Direction.LEFT:
            return player.position[0] - 1, player.position[1]
        elif direction == Direction.UP_LEFT:
            return player.position[0] - 1, player.position[1] - 1
        return -1, -1

    def is_valid_move(self, player: Player, direction: Direction) -> bool:
        # Player can only move to position on grid
        position = self.get_position_after_move(player, direction)
        if position[0] < 0 or position[0] >= GRID_SIZE or position[1] < 0 or position[1] >= GRID_SIZE:
            return False

        # Cop can only move to empty position or mr x position
        if not player.is_mr_x():
            for cop in self.get_cops():
                if cop == player:
                    continue
                if cop.position == position:
                    return False
        #Mr X can only move to empty position
        else:
            for cop in self.get_cops():
                if cop.position == position:
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
        if self.round_number >= MAX_NUMBER_OF_TURNS:
            return GameStatus.MR_X_WON
        return GameStatus.ONGOING

    def get_player_by_name(self, name: str) -> Player | None:
        for player in self.players:
            if player.name == name:
                return player
        return None

    def log_start_info(self):
        verbose_print("Start positions cops: " + str(self.start_positions_cops), self.verbose)
        verbose_print("Start positions mr x: " + str(self.start_positions_mr_x), self.verbose)
        return self

    def get_distance_between_positions(self, position_1: (int, int), position_2: (int, int)) -> float:
        return math.sqrt((position_1[0] - position_2[0]) ** 2 + (position_1[1] - position_2[1]) ** 2)

    def get_possible_mr_x_positions(self) -> [()]:

        if self.get_mr_x().last_known_position is not None:
            possible_mr_x_positions = self.get_square_radius(
                self.get_mr_x().last_known_position, self.get_number_of_rounds_since_last_reveal()
            )
        else:
            possible_mr_x_positions = []
            for starting_position in self.start_positions_mr_x:
                _possible_mr_x_positions = self.get_square_radius(
                    starting_position,
                    self.get_number_of_rounds_since_last_reveal()
                )
                for position in _possible_mr_x_positions:
                    if position not in possible_mr_x_positions:
                        possible_mr_x_positions.append(position)
        return possible_mr_x_positions

    def set_mrx_algo(self, mr_x_algo):
        self.mrx_algorithm = mr_x_algo
        return self

    def set_cop_algo(self, cop_algo):
        self.cop_algorithm = cop_algo
        return self

    #TODO delete
    def get_rewards_fake(self, invalid_actions_player: str | None = None):
        pass
        if invalid_actions_player is not None:
            return {invalid_actions_player: -20}
        minimum_distance = MAX_DISTANCE // 2
        rewards = {agent: 0.0 for agent in self._agent_ids}
        win_rewards = {agent: 0.0 for agent in self._agent_ids}
        inactivity_penalty = {agent: 0.0 for agent in self._agent_ids}
        distance_rewards = {agent: 0.0 for agent in self._agent_ids}
        inside_rewards = {agent: 0.0 for agent in self._agent_ids}

        for agent in self._agent_ids:
            if self.agents_is_in_previous_location_count[agent] > 5:
                inactivity_penalty[agent] = self.agents_is_in_previous_location_count[agent] * (-0.5)
            else:
                inactivity_penalty[agent] = 0

        # __ MR X __ #
        # Distance to cops
        for cop in self.get_cops():
            distance = self.get_mr_x().get_distance_to(cop.position)
            if distance > 0:
                distance_rewards["mr_x"] = round(
                    (((distance - minimum_distance) / MAX_DISTANCE) * 20) / 3, 10)

        # __ COPS __ #

        possible_mr_x_positions = self.get_possible_mr_x_positions()
        if self.get_game_status() == GameStatus.COPS_WON:
            for cop in self.get_cops():
                distance_to_mr_x = cop.get_distance_to(self.get_mr_x().position)
                if distance_to_mr_x == 0:
                    win_rewards[cop.name] = 50
                else:
                    rewards[cop.name] = 30
        elif self.get_game_status() == GameStatus.MR_X_WON:
            win_rewards["mr_x"] = 50
            win_rewards["cop_1"] = -50
            win_rewards["cop_2"] = -50
            win_rewards["cop_3"] = -50

        for cop in self.get_cops():
            # Distance to last known position of mr x

            closest_position = self.get_closest_position(
                cop.position,
                possible_mr_x_positions
            )
            distance_to_closest_position = cop.get_distance_to(closest_position)
            if cop.position in possible_mr_x_positions:
                # Being inside the area of interest is more beneficial
                inside_rewards[cop.name] = 20
            else:
                # Closer to the area of interest, the more reward is gained
                # Maximum reward is 10, so being inside location of interest is more beneficial
                distance_rewards[cop.name] = (float(
                    minimum_distance - distance_to_closest_position) / MAX_DISTANCE) * 10

        for agent in self._agent_ids:
            rewards[agent] = (
                    distance_rewards[agent]
                    + win_rewards[agent]
                    + inactivity_penalty[agent]
                    + inside_rewards[agent]
            )
        return rewards

    # -- END: HELPER FUNCTIONS -- #
