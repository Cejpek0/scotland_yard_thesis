import math
import random
import sys
import time
from enum import Enum
from typing import List

import numpy as np
import pygame
import ray
from gymnasium import spaces
from ray.rllib.algorithms import DQNConfig, DQN
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.examples.models.centralized_critic_models import (
    TorchCentralizedCriticModel,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

from src.Cop import Cop
from src.MrX import MrX
from src.Player import Player
from src.colors import *
from src.environments.rlib.FakeEnv import FakeEnv

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
    def __init__(self, training=False, algorithm_to_use=DefinedAlgorithms.PPO):
        self.number_of_cops = 3
        self.round_number = 0
        self.start_positions_mr_x = []
        self.start_positions_cops = []
        self.grid_size = GRID_SIZE
        self.number_of_starting_positions_cops = NUMBER_OF_STARTING_POSITIONS_COPS
        self.number_of_starting_positions_mr_x = NUMBER_OF_STARTING_POSITIONS_MR_X
        self.players = []
        self.algorithm = None
        self.playing_player_index = 0

        self.agents_previous_locations = {}
        self.agents_is_in_previous_location_count = {}
        self.agents_previous_locations["mr_x"] = None
        self.agents_previous_locations["cop_1"] = None
        self.agents_previous_locations["cop_2"] = None
        self.agents_previous_locations["cop_3"] = None
        self.agents_is_in_previous_location_count["mr_x"] = 0
        self.agents_is_in_previous_location_count["cop_1"] = 0
        self.agents_is_in_previous_location_count["cop_2"] = 0
        self.agents_is_in_previous_location_count["cop_3"] = 0

        if not training:
            def env_creator(env_config):
                return FakeEnv({})

            register_env("scotland_env", env_creator)

            ModelCatalog.register_custom_model(
                "cc_model",
                TorchCentralizedCriticModel
            )

            my_config = (PPOConfig()
                         .training(model={"custom_model": "cc_model"}))

            my_config["policies"] = {
                "mr_x_policy": MR_X_POLICY_SPEC,
                "cop_policy": COP_POLICY_SPEC,
            }

            def policy_mapping_fn(agent_id, episode, worker):
                return "mr_x_policy" if agent_id == "mr_x" else "cop_policy"

            my_config["policy_mapping_fn"] = policy_mapping_fn

            print("3")

            if False:
                from tune_ppo import get_latest_checkpoint
                latest_checkpoint_dir = get_latest_checkpoint()

                self.algorithm = Algorithm.from_checkpoint(latest_checkpoint_dir)
            else:

                if algorithm_to_use == DefinedAlgorithms.PPO:
                    algo = PPO(env="scotland_env", config=my_config)
                    algo.train()
                    algo.restore("trained_policies")
                    self.algorithm = algo
                elif algorithm_to_use == DefinedAlgorithms.DQN:
                    my_config = (DQNConfig()
                                 .training(model={"fcnet_hiddens": [32, 32, 16]},
                                           lr=0.001,
                                           gamma=0.99,
                                           target_network_update_freq=10,
                                           double_q=True,
                                           dueling=True,
                                           num_atoms=1,
                                           noisy=True,
                                           n_step=3, )
                                 .rollouts(observation_filter="MeanStdFilter"))

                    repeat = 50

                    replay_config = {
                        "_enable_replay_buffer_api": True,
                        "type": "MultiAgentPrioritizedReplayBuffer",
                        "capacity": 60000,
                        "prioritized_replay_alpha": 0.5,
                        "prioritized_replay_beta": 0.5,
                        "prioritized_replay_eps": 3e-6,
                    }

                    exploration_config = {
                        "type": "EpsilonGreedy",
                        "initial_epsilon": 1.0,
                        "final_epsilon": 0.05,
                        "epsilon_timesteps": repeat
                    }

                    my_config["replay_config"] = replay_config
                    my_config["exploration_config"] = exploration_config

                    my_config.evaluation_config = {
                        "evaluation_interval": 10,
                        "evaluation_num_episodes": 10,
                    }

                    my_config["policies"] = {
                        "mr_x_policy": MR_X_POLICY_SPEC,
                        "cop_policy": COP_POLICY_SPEC,
                    }

                    def adjust_rollout_fragment_length(iteration, start_length, max_length, total_iterations):
                        progress = iteration / total_iterations
                        return int(start_length + progress * (max_length - start_length))

                    def policy_mapping_fn(agent_id, episode, worker):
                        return "mr_x_policy" if agent_id == "mr_x" else "cop_policy"

                    my_config["policy_mapping_fn"] = policy_mapping_fn

                    my_config["num_rollout_workers"] = 4
                    my_config["reuse_actors"] = True
                    my_config.resources(num_gpus=1, num_gpus_per_worker=0.2)
                    my_config["rollout_fragment_length"] = 500
                    my_config.framework("torch")

                    # Set the config object's env.
                    algo = DQN(env="scotland_env", config=my_config)
                    algo.restore("trained_policies_dqn")
                    self.algorithm = algo
        # Create players
        self.create_players()
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
        self.start_positions_mr_x.clear()
        self.start_positions_cops.clear()
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

        return self

    def quit(self):
        ray.shutdown()
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
        cop_1 = self.get_cop_by_number(1)
        cop_2 = self.get_cop_by_number(2)
        cop_3 = self.get_cop_by_number(3)
        mr_x = self.get_mr_x()

        # mr_x
        if mr_x.last_known_position is not None:
            obs_list_mrx = np.array([
                self.get_current_round_number(),
                self.get_round_turns(),
                self.get_next_reveal_round_number(),
                mr_x.position[0],
                mr_x.position[1],
                mr_x.last_known_position[0],
                mr_x.last_known_position[1],
                mr_x.get_distance_to(mr_x.last_known_position),
                cop_1.position[0],
                cop_1.position[1],
                mr_x.get_distance_to(cop_1.position),
                cop_2.position[0],
                cop_2.position[1],
                mr_x.get_distance_to(cop_2.position),
                cop_3.position[0],
                cop_3.position[1],
                mr_x.get_distance_to(cop_3.position)
            ]).astype(np.float32)
        else:
            obs_list_mrx = np.array([
                self.get_current_round_number(),
                self.get_round_turns(),
                self.get_next_reveal_round_number(),
                mr_x.position[0],
                mr_x.position[1],
                -1,
                -1,
                -1,
                cop_1.position[0],
                cop_1.position[1],
                mr_x.get_distance_to(cop_1.position),
                cop_2.position[0],
                cop_2.position[1],
                mr_x.get_distance_to(cop_2.position),
                cop_3.position[0],
                cop_3.position[1],
                mr_x.get_distance_to(cop_3.position)
            ]).astype(np.float32)

        # cops
        cops_observations = []

        possible_mr_x_positions = self.get_possible_mr_x_positions()

        for cop_number in range(1, self.number_of_cops + 1):
            cop = self.get_cop_by_number(cop_number)
            if mr_x.last_known_position is not None:
                obs_list_cop = np.array([
                    self.get_current_round_number(),
                    self.get_round_turns(),
                    self.get_next_reveal_round_number(),
                    cop.position[0],
                    cop.position[1],
                    mr_x.last_known_position[0],
                    mr_x.last_known_position[1],
                    cop.get_distance_to(mr_x.last_known_position),
                ]).astype(np.float32)
            else:
                obs_list_cop = np.array([
                    self.get_current_round_number(),
                    self.get_round_turns(),
                    self.get_next_reveal_round_number(),
                    cop.position[0],
                    cop.position[1],
                    -1,
                    -1,
                    -1,
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
        player = self.get_player_by_name(player.name)
        while not action_is_valid:
            if count < 100:
                generated_action = self.algorithm.compute_single_action(observations[player.name],
                                                                        policy_id="mr_x_policy" if player.number == 0
                                                                        else "cop_policy")
                print(generated_action)
            else:
                generated_action = self.get_random_action()
                print(f"Generated random action after 100 tries")
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
        # print(f"Player {player.position} moves {direction}")
        if self.is_valid_move(player, direction):
            player.position = self.get_position_after_move(player, direction)
        else:
            print(f"{player.name} tried to move {direction} from {player.position}")
            print(f"Would result in {self.get_position_after_move(player, direction)}")
            sys.stderr.write(f"Move {direction} is not valid\n")
            exit(1)
        return self

    def get_current_player(self):
        return self.players[self.playing_player_index]

    def play_turn(self, cop_algo=DefinedAlgorithms.PPO, mr_x_algo=DefinedAlgorithms.PPO):
        if self.playing_player_index == 0:
            self.round_number += 1

        player = self.get_current_player()
        action = None
        if player.is_mr_x():
            if mr_x_algo == DefinedAlgorithms.PPO:
                action = self.get_action_for_player(player)
            elif mr_x_algo == DefinedAlgorithms.DQN:
                action = self.get_action_for_player(player)
            else:
                action = self.get_random_action()
        elif player.is_cop():
            if cop_algo == DefinedAlgorithms.PPO:
                action = self.get_action_for_player(player)
            elif cop_algo == DefinedAlgorithms.DQN:
                action = self.get_action_for_player(player)
            else:
                action = self.get_random_action()

        self.move_player(player, action)
        if self.round_number in REVEAL_POSITION_ROUNDS and player.is_mr_x():
            self.get_mr_x().mr_x_reveal_position()
        self.playing_player_index = (self.playing_player_index + 1) % len(self.players)

        # Display rewards for each agent
        rewards = self.get_rewards_fake()
        print("Rewards:")
        for player in self.players:
            print(f"{player.name}: {rewards[player.name]}")

        return self

    # TODO: delete
    def get_rewards_fake(self, invalid_actions_players: List[str] = None):
        if invalid_actions_players is None:
            invalid_actions_players = []
        distance_reward = 0
        inactivity_penalty = 0
        minimum_distance = 5
        rewards = {}

        # __ MR X __ #
        if "mr_x" in invalid_actions_players:
            rewards["mr_x"] = -20
        else:
            if self.get_mr_x().position == self.agents_previous_locations["mr_x"]:
                self.agents_is_in_previous_location_count["mr_x"] += 1
            else:
                self.agents_is_in_previous_location_count["mr_x"] = 0
            self.agents_previous_locations["mr_x"] = self.get_mr_x().position
            inactivity_penalty = self.agents_is_in_previous_location_count["mr_x"] * (-1)
            # Distance to cops
            for cop in self.get_cops():
                distance = self.get_mr_x().get_distance_to(cop.position)
                if distance == 0:
                    distance_reward -= 100
                else:
                    distance_reward += round((distance - minimum_distance), 10)
            rewards["mr_x"] = distance_reward + inactivity_penalty

        # __ COPS __ #

        possible_mr_x_positions = self.get_possible_mr_x_positions()

        if self.get_game_status() == GameStatus.COPS_WON:
            for cop in self.get_cops():
                distance_to_mr_x = cop.get_distance_to(self.get_mr_x().position)
                if distance_to_mr_x == 0:
                    rewards[cop.name] = 100
                else:
                    rewards[cop.name] = 50
            return rewards

        for cop in self.get_cops():
            if f"cop_{cop.number}" in invalid_actions_players:
                rewards[cop.name] = -20
                continue

            if cop.position == self.agents_previous_locations[cop.name]:
                self.agents_is_in_previous_location_count[cop.name] += 1
            else:
                self.agents_is_in_previous_location_count[cop.name] = 0
            self.agents_previous_locations[cop.name] = cop.position

            inactivity_penalty = self.agents_is_in_previous_location_count[cop.name] * (-1)

            # Check winnning condition
            if cop.position == self.get_mr_x().position:
                for cop in self.get_cops():
                    rewards[cop.name] = 100
                continue

            # Distance to last known position of mr x
            distance_reward = 0
            inside_reward = 0

            closest_position = self.get_closest_position(
                cop.position,
                possible_mr_x_positions
            )
            distance_to_closest_position = cop.get_distance_to(closest_position)
            if cop.position in possible_mr_x_positions:
                # Being inside the area of interest is more beneficial
                inside_reward = 10
            else:
                # Closer to the area of interest, the more reward is gained
                # Maximum reward is 5, so being inside location of interest is more beneficial
                inside_reward = (
                        ((minimum_distance - distance_to_closest_position) / MAX_DISTANCE) * 5)

            total_reward = distance_reward + inside_reward + inactivity_penalty
            rewards[cop.name] = total_reward
        return rewards

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

    def get_cop_by_number(self, number: int) -> Player | None:
        for cop in self.get_cops():
            if cop.number == number:
                return cop
        return None

    def get_current_round_number(self):
        return self.round_number

    def get_round_turns(self):
        return MAX_NUMBER_OF_TURNS

    def get_next_reveal_round_number(self):
        for turn_number in REVEAL_POSITION_ROUNDS:
            if turn_number > self.round_number:
                return turn_number
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
        if player.number != 0:
            for cop in self.get_cops():
                if cop == player:
                    continue
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
        print("Start positions cops: " + str(self.start_positions_cops))
        print("Start positions mr x: " + str(self.start_positions_mr_x))
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
    # -- END: HELPER FUNCTIONS -- #
