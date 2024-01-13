from typing import Dict, List, Any

import numpy as np
from gymnasium import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import AgentID
import random

import scotland_yard_game

_agent_ids = ["mr_x", "cop"]


class ScotlandYardEnvironment1v1(MultiAgentEnv):
    _agent_ids = ["mr_x", "cop"]

    def __init__(self, config):
        scotland_yard_game.NUMBER_OF_COPS = 1
        super().__init__()
        self.config = config
        self.game = scotland_yard_game.ScotlandYard(training=True, number_of_cops=1)
        self.observations = None
        self._agent_ids = ["mr_x", "cop_1"]
        self.num_agents = len(self._agent_ids)

        self.action_space = spaces.dict.Dict({"mr_x": spaces.Discrete(4), "cop_1": spaces.Discrete(4)})

        self.observation_space = spaces.dict.Dict({
            "mr_x": spaces.Box(low=np.array([
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
            ]), high=np.array([
                scotland_yard_game.MAX_NUMBER_OF_TURNS,  # current turn
                scotland_yard_game.MAX_NUMBER_OF_TURNS,  # max turns
                scotland_yard_game.MAX_NUMBER_OF_TURNS,  # next reveal
                scotland_yard_game.GRID_SIZE,  # position x
                scotland_yard_game.GRID_SIZE,  # position y
                scotland_yard_game.GRID_SIZE,  # position x of cop
                scotland_yard_game.GRID_SIZE,  # position y or cop
                scotland_yard_game.GRID_SIZE,  # last known position x
                scotland_yard_game.GRID_SIZE,  # last known position y
                scotland_yard_game.GRID_SIZE * 2  # distance to cop
            ]), dtype=np.float32),
            "cop_1": spaces.Box(low=np.array([
                0,  # current turn
                0,  # max turns
                0,  # next reveal
                0,  # position x
                0,  # position y
                -1,  # last known position x
                -1,  # last known position y
            ]), high=np.array([
                scotland_yard_game.MAX_NUMBER_OF_TURNS,  # current turn
                scotland_yard_game.MAX_NUMBER_OF_TURNS,  # max turns
                scotland_yard_game.MAX_NUMBER_OF_TURNS,  # next reveal
                scotland_yard_game.GRID_SIZE,  # position x
                scotland_yard_game.GRID_SIZE,  # position y
                scotland_yard_game.GRID_SIZE,  # last known position x
                scotland_yard_game.GRID_SIZE,  # last known position y
            ]), dtype=np.float32)
        })

    def step(self, action_dict: Dict[AgentID, int]) -> (
            Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]):
        # check if actions of all agents are valid
        invalid_actions_players = []
        for agent_id, action in action_dict.items():
            if agent_id == "mr_x":
                if scotland_yard_game.Direction(action) not in self.game.get_players_valid_moves(
                        self.game.get_mr_x()):
                    invalid_actions_players.append(agent_id)
            else:
                if scotland_yard_game.Direction(action) not in self.game.get_players_valid_moves(
                        self.game.get_cops()[0]):
                    invalid_actions_players.append(agent_id)

        # If any action is invalid, do not proceed with the step. Do not update the game, and heavily penalize agents
        # that made invalid actions
        if len(invalid_actions_players) > 0:
            return self.get_observations(), self.get_rewards(invalid_actions_players), self.getTerminations(),{"__all__": False, "mr_x": False, "cop_1": False}, {"mr_x": {}, "cop_1": {}}

        # Proceed with the step

        # Reveal mr x position if current turn is in REVEAL_POSITION_TURNS
        if self.game.get_current_turn_number() in scotland_yard_game.REVEAL_POSITION_TURNS:
            self.game.get_mr_x().last_known_position = self.game.get_mr_x().position

        # Move players
        for agent_id, action in action_dict.items():
            if agent_id == "mr_x":
                self.game.move_player(self.game.get_mr_x(), scotland_yard_game.Direction(action))
            else:
                self.game.move_player(self.game.get_cops()[0], scotland_yard_game.Direction(action))

        # Update game state
        self.game.turn_number += 1

        # Get updated observations
        observations = self.get_observations()

        # Calculate rewards
        rewards = self.get_rewards()

        # Check if the game is over
        terminates = self.getTerminations()

        # Return observations, rewards, terminates,truncateds, info
        return observations, rewards, terminates, {"__all__": False, "mr_x": False, "cop_1": False}, {"mr_x": {},"cop_1": {}}

    def reset(self, *, seed=None, options=None):
        self.game.reset()
        self.game.start_positions_cops = self.game.generate_start_positions([],
                                                                            self.game.number_of_starting_positions_cops)
        self.game.start_positions_mr_x = self.game.generate_start_positions(self.game.start_positions_cops,
                                                                            self.game.number_of_starting_positions_mr_x)
        self.game.get_mr_x().set_start_position(
            self.game.start_positions_mr_x[random.randint(0, len(self.game.start_positions_mr_x) - 1)])
        for cop in self.game.get_cops():
            cop.set_start_position(
                self.game.start_positions_cops[random.randint(0, len(self.game.start_positions_cops) - 1)])
        # Observations
        observations = self.get_empty_observation()
        return observations, {"mr_x": 0, "cop_1": 0}

    def get_rewards(self, invalid_actions_players: List[str] = None):
        if invalid_actions_players is None:
            invalid_actions_players = []
        distance_reward = 0
        minimum_distance = 5
        rewards = {}
        # __ MR X __ #
        if "mr_x" in invalid_actions_players:
            rewards["mr_x"] = -100
        else:
            # Distance to cops
            for cop in self.game.get_cops():
                distance = self.game.get_mr_x().get_distance_to(cop)
                if distance == 0:
                    distance_reward -= 100
                else:
                    distance_reward += round((distance - minimum_distance) * (1 / 3), 10)
            # Distance to last known position
            if self.game.get_mr_x().last_known_position is not None:
                distance_reward += round(
                    self.game.get_mr_x().get_distance_to(self.game.get_mr_x().last_known_position) * 0.5,
                    10)
            rewards["mr_x"] = distance_reward

        # __ COPS __ #
        for cop in self.game.get_cops():
            if f"cop_{cop.number}" in invalid_actions_players:
                rewards[f"cop_{cop.number}"] = -100
                continue
            # Distance to last known position of mr x
            distance_reward = 0
            if self.game.get_mr_x().last_known_position is None:
                distance_reward = 0
            else:
                distance = self.game.get_mr_x().get_distance_to(cop)
                if distance == 0:
                    distance_reward += 100
                else:
                    distance_to_last_known_position = cop.get_distance_to(self.game.get_mr_x().last_known_position)
                    distance_reward += round((distance_to_last_known_position) * (1 / 3), 10)
            rewards["cop_" + str(cop.number)] = distance_reward

        return rewards

    def get_observations(self):
        # mr_x
        mr_x = self.game.get_mr_x()
        if mr_x.last_known_position is not None:
            obs_list_mrx = np.array([
                self.game.get_current_turn_number(),
                self.game.get_max_turns(),
                self.game.get_next_reveal_turn_number(),
                mr_x.position[0],
                mr_x.position[1],
                self.game.get_cops()[0].position[0],
                self.game.get_cops()[0].position[1],
                mr_x.last_known_position[0],
                mr_x.last_known_position[1],
                mr_x.get_distance_to(self.game.get_cops()[0])
            ]).astype(np.float32)
        else:
            obs_list_mrx = np.array([
                self.game.get_current_turn_number(),
                self.game.get_max_turns(),
                self.game.get_next_reveal_turn_number(),
                mr_x.position[0],
                mr_x.position[1],
                self.game.get_cops()[0].position[0],
                self.game.get_cops()[0].position[1],
                -1,
                -1,
                mr_x.get_distance_to(self.game.get_cops()[0])
            ]).astype(np.float32)

        # cops
        cop = self.game.get_cops()[0]

        if mr_x.last_known_position is not None:
            obs_list_cop = np.array([
                self.game.get_current_turn_number(),
                self.game.get_max_turns(),
                self.game.get_next_reveal_turn_number(),
                cop.position[0],
                cop.position[1],
                mr_x.last_known_position[0],
                mr_x.last_known_position[1],
            ]).astype(np.float32)
        else:
            obs_list_cop = np.array([
                self.game.get_current_turn_number(),
                self.game.get_max_turns(),
                self.game.get_next_reveal_turn_number(),
                cop.position[0],
                cop.position[1],
                -1,
                -1,
            ]).astype(np.float32)

        observations = {
            "mr_x": obs_list_mrx,
            "cop_1": obs_list_cop
        }

        return observations

    def get_empty_observation(self):
        mrx_obs = np.zeros(10)
        mrx_obs[7] = -1
        mrx_obs[8] = -1
        mrx_obs = mrx_obs.astype(np.float32)
        cop_obs = np.zeros(7)
        cop_obs[5] = -1
        cop_obs[6] = -1
        cop_obs = cop_obs.astype(np.float32)
        return {
            "mr_x": mrx_obs,
            "cop_1": cop_obs
        }

    def getTerminations(self):
        game_over = self.game.get_game_status()
        returnDict = {}
        for agent_id in self._agent_ids:
            if game_over == scotland_yard_game.GameStatus.ONGOING:
                returnDict[agent_id] = False
            else:
                returnDict[agent_id] = True
        if game_over == scotland_yard_game.GameStatus.ONGOING:
            returnDict["__all__"] = False
        else:
            returnDict["__all__"] = True
        return returnDict
