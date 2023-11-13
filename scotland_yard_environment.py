from typing import Dict, List, Any

import numpy
from gymnasium import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import AgentID
import random

import scotland_yard_game


class ScotlandYardEnvironment(MultiAgentEnv):
    _agent_ids = ["mr_x", "cop_1", "cop_2", "cop_3"]
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.game = scotland_yard_game.ScotlandYard()
        self.num_agents = len(self.game.players)
        self.observations = None
        # Define your environment state and other components here
        self.action_space = dict(mr_x=spaces.Discrete(4))
        for cop in self.game.get_cops():
            self.action_space["cop_" + str(cop.number)] = spaces.Discrete(4)
        mr_x_number_of_observations = len(
            ["current turn", "max turns", "next_reveal", "position: x", "position: y", "last known position: x",
             "last known position: y"])
        for cop in range(len(self.game.get_cops())):
            mr_x_number_of_observations += len(["distance to cop", "position: x", "position: y"])
        cop_number_of_observations = len(
            ["current turn", "max turns", "next_reveal_turn", "position: x", "position: y", "last known position: x",
             "last known position: y"])
        for cop in range(len(self.game.get_cops()) - 1):
            cop_number_of_observations += len(["position: x", "position: y"])
        self.observation_space = {"mr_x": spaces.Box(low=-100, high=100, shape=(mr_x_number_of_observations,))}
        for cop in self.game.get_cops():
            self.observation_space["cop_" + str(cop.number)] = spaces.Box(low=-100,
                                                                     high=100,
                                                                     shape=(cop_number_of_observations,))

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
        observations = self.get_observation()

        # Calculate rewards
        rewards = self.get_rewards()

        # Check if the game is over
        terminates = self.get_terminates()

        return observations, rewards, terminates, {}, {}

    def reset(self, **kwargs):
        self.game.reset()
        self.game.start_positions_cops = self.game.generate_start_positions([], self.game.number_of_starting_positions_cops)
        self.game.start_positions_mr_x = self.game.generate_start_positions(self.game.start_positions_cops, self.game.number_of_starting_positions_mr_x)
        self.game.get_mr_x().set_start_position(
            self.game.start_positions_mr_x[random.randint(0, len(self.game.start_positions_mr_x) - 1)])
        for cop in self.game.get_cops():
            cop.set_start_position(
                self.game.start_positions_cops[random.randint(0, len(self.game.start_positions_cops) - 1)])
        # Observations
        self.observations = self.get_observation()
        return self.observations, {}

    def get_terminates(self):
        game_over = self.game.is_game_over()
        if game_over != scotland_yard_game.GameStatus.ONGOING:
            return {"__all__": True}

    def get_rewards(self):
        rewards = {}
        # __ MR X __ #
        # Distance to cops
        distance_reward = 0
        minimum_distance = 2
        for cop in self.game.get_cops():
            distance = self.game.get_mr_x().get_distance_to(cop)
            if distance == 0:
                distance_reward -= 100
            distance_reward += round((distance - minimum_distance) * (1 / 3), 10)
        # Distance to last known position
        if(self.game.get_mr_x().last_known_position is not None):
            distance_reward += round(self.game.get_mr_x().get_distance_to(self.game.get_mr_x().last_known_position) * 0.5,
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
                    distance_reward += 100
                distance_reward += round((distance - minimum_distance) * 0.1, 10)
            rewards["cop_" + str(cop.number)] = distance_reward

        return rewards

    def get_observation(self):
        observations = {}
        # mr_x
        if self.game.get_mr_x().last_known_position is not None:
            arr = [self.game.get_current_turn_number(), self.game.get_max_turns(), self.game.get_next_reveal_turn_number(),
                   self.game.get_mr_x().position[0], self.game.get_mr_x().position[1],
                   self.game.get_mr_x().last_known_position[0], self.game.get_mr_x().last_known_position[1]]
        else:
            arr = [self.game.get_current_turn_number(), self.game.get_max_turns(), self.game.get_next_reveal_turn_number(),
                   self.game.get_mr_x().position[0], self.game.get_mr_x().position[1], None, None]
        for cop in self.game.get_cops():
            arr.append(self.game.get_mr_x().get_distance_to(cop.position))
            arr.append(cop.position[0])
            arr.append(cop.position[1])
        observations["mr_x"] = arr

        # cops
        for cop in self.game.get_cops():
            if(self.game.get_mr_x().last_known_position is not None):
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
        return observations
