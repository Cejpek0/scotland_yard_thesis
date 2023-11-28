from typing import Dict, List, Any

import numpy
from gymnasium import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict, MultiEnvDict
import random

import scotland_yard_game

_agent_ids = ["mr_x", "cop"]


class ScotlandYardEnvironment1v1(MultiAgentEnv):
    _agent_ids = ["mr_x", "cop"]

    def __init__(self, config):
        scotland_yard_game.NUMBER_OF_COPS = 1
        super().__init__()
        self.config = config
        self.game = scotland_yard_game.ScotlandYard()
        self.num_agents = len(self.game.players)
        self.observations = None
        self._spaces_in_preferred_format = True
        self._agent_ids = ["mr_x", "cop_1"]

        self.action_spaces = {"mr_x": spaces.Discrete(4), "cop_1": spaces.Discrete(4)}

        mr_x_number_of_observations = len(
            ["current turn", "max turns", "next_reveal", "position: x", "position: y", "last known position: x",
             "last known position: y"])
        for cop in range(len(self.game.get_cops())):
            mr_x_number_of_observations += len(["distance to cop", "position: x", "position: y"])
        cop_number_of_observations = len(
            ["current turn", "max turns", "next_reveal_turn", "position: x", "position: y", "last known position: x",
             "last known position: y"])
        self.observation_spaces = {"mr_x": spaces.Box(low=0, high=1, shape=(mr_x_number_of_observations, 0)),
                                   "cop_1": spaces.Box(low=0, high=1, shape=(cop_number_of_observations, 0))}

    def step(self, action_dict: Dict[AgentID, int]) -> (
            Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]):

        if self.game.get_current_turn_number() in scotland_yard_game.REVEAL_POSITION_TURNS:
            self.game.get_mr_x().last_known_position = self.game.get_mr_x().position

        # Execute actions
        for agent_id, action in action_dict.items():
            if agent_id == "mr_x":
                agent = self.game.get_mr_x()
            else:
                agent = self.game.get_cop_by_number(int(str.replace(agent_id, "cop_", "")))
            if agent is not None:
                agent.move(scotland_yard_game.Direction(action))

        # Update game state
        self.game.turn_number += 1

        # Get updated observations
        observations = self.get_observations()

        # Calculate rewards
        rewards = self.get_rewards()

        # Check if the game is over
        terminates = self.get_terminates()

        return observations, rewards, terminates, {}, {}

    def reset(self, **kwargs):
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
        self.observations = self.get_observations()
        return self.observations, {}

    def get_terminates(self):
        game_over = self.game.is_game_over()
        returnDict = {}
        for agent_id in self._agent_ids:
            if game_over == scotland_yard_game.GameStatus.ONGOING:
                returnDict[agent_id] = False
            else:
                returnDict[agent_id] = True
        if game_over != scotland_yard_game.GameStatus.ONGOING:
            returnDict["__all__"] = True
        else:
            returnDict["__all__"] = False
        return returnDict

    def get_rewards(self):
        rewards = {}
        # __ MR X __ #
        # Distance to cops
        distance_reward = 0
        minimum_distance = 2
        for cop in self.game.get_cops():
            distance = self.game.get_mr_x().get_distance_to(cop)
            if distance == 0:
                distance_reward -= 1
            distance_reward += round((distance - minimum_distance) * (1 / 3), 10)
        # Distance to last known position
        if (self.game.get_mr_x().last_known_position is not None):
            distance_reward += round(
                self.game.get_mr_x().get_distance_to(self.game.get_mr_x().last_known_position) * 0.5,
                10)
        rewards["mr_x"] = distance_reward

        # __ COPS __ #
        for cop in self.game.get_cops():
            # Distance to last known position of mr x
            distance_reward = 0
            if self.game.get_mr_x().last_known_position is None:
                distance_reward = 0
            else:
                distance = self.game.get_mr_x().last_known_position.get_distance_to(cop)
                if distance == 0:
                    distance_reward += 1
                distance_reward += round((distance - minimum_distance) * 0.1, 10)
            rewards["cop_" + str(cop.number)] = distance_reward

        return rewards

    def get_observations(self):
        observations = {}

        # mr_x
        if self.game.get_mr_x().last_known_position is not None:
            arr = [(self.game.get_current_turn_number()), self.game.get_max_turns(),
                   self.game.get_next_reveal_turn_number(),
                   self.game.get_mr_x().position[0], self.game.get_mr_x().position[1],
                   self.game.get_mr_x().last_known_position[0], self.game.get_mr_x().last_known_position[1]]
        else:
            arr = [self.game.get_current_turn_number(), self.game.get_max_turns(),
                   self.game.get_next_reveal_turn_number(),
                   self.game.get_mr_x().position[0], self.game.get_mr_x().position[1], None, None]
        for cop in self.game.get_cops():
            arr.append(self.game.get_mr_x().get_distance_to(cop.position))
            arr.append(cop.position[0])
            arr.append(cop.position[1])
        observations["mr_x"] = arr

        # cops
        for cop in self.game.get_cops():
            if (self.game.get_mr_x().last_known_position is not None):
                arr = [self.game.get_current_turn_number(), self.game.get_max_turns(),
                       self.game.get_next_reveal_turn_number(),
                       cop.position[0], cop.position[1], self.game.get_mr_x().last_known_position[0],
                       self.game.get_mr_x().last_known_position[1]]
            else:
                arr = [self.game.get_current_turn_number(), self.game.get_max_turns(),
                       self.game.get_next_reveal_turn_number(),
                       cop.position[0], cop.position[1], None, None]
            for otherCop in self.game.get_cops():
                if cop != otherCop:
                    arr.append(otherCop.position[0])
                    arr.append(otherCop.position[1])
            observations["cop_" + str(cop.number)] = arr

        observations["mr_x"] = numpy.array(observations["mr_x"]).astype(numpy.float32)
        observations["cop_1"] = numpy.array(observations["cop_1"]).astype(numpy.float32)
        return observations

    def observation_space_sample(self, agent_ids: list = None) -> MultiEnvDict:
        if agent_ids is None:
            agent_ids = self._agent_ids
        return {agent_id: self.observation_spaces[agent_id].sample() for agent_id in agent_ids}

    def action_space_sample(self, agent_ids: list = None) -> MultiEnvDict:
        if agent_ids is None:
            agent_ids = self._agent_ids
        return {agent_id: self.action_spaces[agent_id].sample() for agent_id in agent_ids}

    def observation_space_contains(self, observation: MultiEnvDict, agent_ids: list = None) -> bool:
        if agent_ids is None:
            agent_ids = self._agent_ids
        return all(self.observation_spaces[agent_id].contains(observation[agent_id]) for agent_id in agent_ids)

    def action_space_contains(self, action: MultiEnvDict, agent_ids: list = None) -> bool:
        if agent_ids is None:
            agent_ids = self._agent_ids
        return all(self.action_spaces[agent_id].contains(action[agent_id]) for agent_id in agent_ids)
