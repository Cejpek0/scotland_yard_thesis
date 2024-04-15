from typing import Dict, List, Any

import numpy as np
from gymnasium import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import AgentID
import random

import src.game.scotland_yard_game_logic as scotland_yard_game


class ScotlandYardEnvironment(MultiAgentEnv):
    _agent_ids = ["mr_x", "cop_1", "cop_2", "cop_3"]

    def __init__(self, config, selected_algo):
        self.next_agent_index = 0
        self.next_agent = "cop_1"
        super().__init__()
        self.config = config
        self.game = scotland_yard_game.ScotlandYardGameLogic(True, selected_algo, selected_algo)
        self.observations = None
        self._agent_ids = ["mr_x", "cop_1", "cop_2", "cop_3"]
        self.agents = self._agent_ids
        self.num_agents = len(self._agent_ids)

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
        current_agent = self.game.get_current_player()
        print(action_dict)
        print(current_agent.name)
        action = action_dict[current_agent.name]

        direction = scotland_yard_game.Direction(action)

        invalid_move = False

        if self.game.is_valid_move(current_agent, direction) is False:
            invalid_move = True

        # If any action is invalid, do not proceed with the step. Do not update the game, and heavily penalize agents
        # that made invalid actions
        if invalid_move:
            observations = {current_agent.name: self.get_observations()[current_agent.name]}
            infos = {current_agent.name: {}}

            return observations, self.get_rewards([current_agent.name]), self.getTerminations(), {
                "__all__": False, "mr_x": False, "cop_1": False, "cop_2": False, "cop_3": False, }, infos

        self.game.play_turn(direction)

        # Get updated observations
        next_player = self.game.get_player_by_number((current_agent.number + 1) % self.num_agents)
        assert next_player is not None
        observations = {next_player.name: self.get_observations()[next_player.name]}
        infos = {next_player.name: {}}

        # Calculate rewards
        rewards = self.get_rewards()

        # Check if the game is over
        terminates = self.getTerminations()

        # Return observations, rewards, terminates,truncated, info
        return observations, rewards, terminates, {"__all__": False, "mr_x": False, "cop_1": False, "cop_2": False,
                                                   "cop_3": False, }, infos

    def reset(self, *, seed=None, options=None):
        self.game.reset()
        current_player = self.game.get_current_player()
        # Observations
        observations = {current_player.name: self.get_observations()[current_player.name]}

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
            if self.game.agents_is_in_previous_location_count["mr_x"] > 5:
                inactivity_penalty = self.game.agents_is_in_previous_location_count["mr_x"] * (-1)
            else:
                inactivity_penalty = 0
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

            if self.game.agents_is_in_previous_location_count[cop.name] > 5:
                inactivity_penalty = self.game.agents_is_in_previous_location_count[cop.name] * (-1)
            else:
                inactivity_penalty = 0
            # Check winning condition
            if cop.position == self.game.get_mr_x().position:
                for cop in self.game.get_cops():
                    rewards[cop.name] = 100
                continue

            # Distance to last known position of mr x
            distance_reward = 0

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
        mrx_obs = np.array([0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        mrx_obs = mrx_obs.astype(np.float32)

        cop_obs = np.array([0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cop_obs = cop_obs.astype(np.float32)

        return {
            "mr_x": mrx_obs,
            "cop_1": cop_obs,
            "cop_2": cop_obs,
            "cop_3": cop_obs
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
