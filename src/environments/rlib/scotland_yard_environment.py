from typing import Dict, List, Any

import numpy as np
from gymnasium import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import AgentID
import random

import src.game.scotland_yard_game_logic as scotland_yard_game


class ScotlandYardEnvironment(MultiAgentEnv):
    _agent_ids = ["mr_x", "cop_1", "cop_2", "cop_3"]

    def __init__(self, config):
        self.next_agent_index = 1
        self.next_agent = "cop_1"
        super().__init__()
        self.config = config
        self.game = scotland_yard_game.ScotlandYardGameLogic(training=True)
        self.observations = None
        self._agent_ids = ["mr_x", "cop_1", "cop_2", "cop_3"]
        self.agents = self._agent_ids
        self.num_agents = len(self._agent_ids)
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

        self.action_space = spaces.dict.Dict({
            "mr_x": spaces.Discrete(9),
            "cop_1": spaces.Discrete(9),
            "cop_2": spaces.Discrete(9),
            "cop_3": spaces.Discrete(9)
        })

        self.observation_space = spaces.dict.Dict({
            "mr_x": scotland_yard_game.mrx_observation_space,
            "cop_1": scotland_yard_game.general_cop_observation_space,
            "cop_2": scotland_yard_game.general_cop_observation_space,
            "cop_3": scotland_yard_game.general_cop_observation_space
        })

    def step(self, action_dict: Dict[AgentID, int]) -> (
            Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]):
        # check if actions of all agents are valid

        if len(action_dict) == 0:
            exit(1)
        agent_id = next(iter(action_dict.keys()))
        action = next(iter(action_dict.values()))

        invalid_move = False

        if scotland_yard_game.Direction(action) not in self.game.get_players_valid_moves(
                self.game.get_player_by_name(agent_id)):
            invalid_move = True

        # If any action is invalid, do not proceed with the step. Do not update the game, and heavily penalize agents
        # that made invalid actions
        if invalid_move:
            observations = {agent_id: self.get_observations()[agent_id]}
            infos = {agent_id: {}}

            return observations, self.get_rewards(agent_id), self.getTerminations(), {
                "__all__": False, "mr_x": False, "cop_1": False, "cop_2": False, "cop_3": False, }, infos

        # Proceed with the step

        # Move player
        self.game.move_player(self.game.get_player_by_name(agent_id), scotland_yard_game.Direction(action))

        # Get updated observations
        observations = {self.next_agent: self.get_observations()[self.next_agent]}
        infos = {self.next_agent: {}}
        self.next_agent = self.agents[(self.next_agent_index + 1) % self.num_agents]
        self.next_agent_index = (self.next_agent_index + 1) % self.num_agents
        # Calculate rewards
        rewards = self.get_rewards()

        # Check if the game is over
        terminates = self.getTerminations()

        # Update game state
        if "cop_3" in action_dict.keys():
            self.game.round_number += 1
            # Reveal mr x position if current turn is in REVEAL_POSITION_ROUNDS
            if self.game.get_current_round_number() in scotland_yard_game.REVEAL_POSITION_ROUNDS:
                self.game.get_mr_x().last_known_position = self.game.get_mr_x().position

        # Return observations, rewards, terminates,truncated, info
        return observations, rewards, terminates, {"__all__": False, "mr_x": False, "cop_1": False, "cop_2": False,
                                                   "cop_3": False, }, infos

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
        self.next_agent_index = 1
        self.next_agent = "cop_1"

        return observations, {"mr_x": 0, "cop_1": 0, "cop_2": 0, "cop_3": 0}

    def get_rewards(self, invalid_actions_players: List[str] = None):
        if invalid_actions_players is None:
            invalid_actions_players = []
        distance_reward = 0
        minimum_distance = 5
        rewards = {}

        # __ MR X __ #
        if "mr_x" in invalid_actions_players:
            rewards["mr_x"] = -20
        else:
            if self.game.get_mr_x().position == self.agents_previous_locations["mr_x"]:
                self.agents_is_in_previous_location_count["mr_x"] += 1
            else:
                self.agents_is_in_previous_location_count["mr_x"] = 0
            self.agents_previous_locations["mr_x"] = self.game.get_mr_x().position
            inactivity_penalty = self.agents_is_in_previous_location_count["mr_x"] * (-1)
            # Distance to cops
            for cop in self.game.get_cops():
                distance = self.game.get_mr_x().get_distance_to(cop.position)
                if distance == 0:
                    distance_reward -= 100
                else:
                    distance_reward += round((distance - minimum_distance), 10)
            rewards["mr_x"] = distance_reward + inactivity_penalty

        # __ COPS __ #

        possible_mr_x_positions = self.game.get_possible_mr_x_positions()

        if self.game.get_game_status() == scotland_yard_game.GameStatus.COPS_WON:
            for cop in self.game.get_cops():
                distance_to_mr_x = cop.get_distance_to(self.game.get_mr_x().position)
                if distance_to_mr_x == 0:
                    rewards[cop.name] = 100
                else:
                    rewards[cop.name] = 50
            return rewards

        for cop in self.game.get_cops():
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
            if cop.position == self.game.get_mr_x().position:
                for cop in self.game.get_cops():
                    rewards[cop.name] = 100
                continue

            # Distance to last known position of mr x
            distance_reward = 0
            inside_reward = 0

            closest_position = self.game.get_closest_position(
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
                        ((minimum_distance - distance_to_closest_position) / scotland_yard_game.MAX_DISTANCE) * 5)

            total_reward = distance_reward + inside_reward + inactivity_penalty
            rewards[cop.name] = total_reward
        return rewards

    def get_observations(self):
        observations = self.game.get_observations()
        return observations

    def get_empty_observation(self):
        mrx_obs = np.array([0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        mrx_obs = mrx_obs.astype(np.float32)
        return {
            "mr_x": mrx_obs
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
