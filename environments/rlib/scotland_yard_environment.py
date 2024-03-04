from typing import Dict, List, Any

import numpy as np
from gymnasium import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import AgentID
import random


class ScotlandYardEnvironment(MultiAgentEnv):
    _agent_ids = ["mr_x", "cop_1", "cop_2", "cop_3"]

    def __init__(self, config):
        self.next_agent_index = 1
        self.next_agent = "cop_1"
        scotland_yard_game.NUMBER_OF_COPS = 1
        super().__init__()
        self.config = config
        self.game = scotland_yard_game.ScotlandYard(training=True)
        self.observations = None
        self._agent_ids = ["mr_x", "cop_1", "cop_2", "cop_3"]
        self.agents = self._agent_ids
        self.num_agents = len(self._agent_ids)

        self.action_space = spaces.dict.Dict({
            "mr_x": spaces.Discrete(4),
            "cop_1": spaces.Discrete(4),
            "cop_2": spaces.Discrete(4),
            "cop_3": spaces.Discrete(4)
        })

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
        ]), high=np.array([
            scotland_yard_game.MAX_NUMBER_OF_TURNS,  # current turn
            scotland_yard_game.MAX_NUMBER_OF_TURNS,  # max turns
            scotland_yard_game.MAX_NUMBER_OF_TURNS,  # next reveal
            scotland_yard_game.GRID_SIZE,  # position x
            scotland_yard_game.GRID_SIZE,  # position y
            scotland_yard_game.GRID_SIZE,  # last known position x of mr x
            scotland_yard_game.GRID_SIZE,  # last known position y of mr x
            scotland_yard_game.MAX_DISTANCE,  # distance to last known position of mr x
            scotland_yard_game.GRID_SIZE,  # position x of other cop_1
            scotland_yard_game.GRID_SIZE,  # position y or other cop_1
            scotland_yard_game.MAX_DISTANCE,  # distance to other cop_1
            scotland_yard_game.GRID_SIZE,  # position x of other cop_2
            scotland_yard_game.GRID_SIZE,  # position y or other cop_2
            scotland_yard_game.MAX_DISTANCE,  # distance to other cop_2,
            scotland_yard_game.GRID_SIZE,  # position x of closest point of interest
            scotland_yard_game.GRID_SIZE,  # position y of closest point of interest
            scotland_yard_game.MAX_DISTANCE,  # distance to the closest point of interest
            1,  # inside or outside area of interest
        ]), dtype=np.float32)

        self.observation_space = spaces.dict.Dict({
            "mr_x": spaces.Box(low=np.array([
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
            ]), high=np.array([
                scotland_yard_game.MAX_NUMBER_OF_TURNS,  # current turn
                scotland_yard_game.MAX_NUMBER_OF_TURNS,  # max turns
                scotland_yard_game.MAX_NUMBER_OF_TURNS,  # next reveal
                scotland_yard_game.GRID_SIZE,  # position x
                scotland_yard_game.GRID_SIZE,  # position y
                scotland_yard_game.GRID_SIZE,  # last known position x
                scotland_yard_game.GRID_SIZE,  # last known position y
                scotland_yard_game.GRID_SIZE * 2,  # distance to last known position
                scotland_yard_game.GRID_SIZE,  # position x of cop_1
                scotland_yard_game.GRID_SIZE,  # position y or cop_1
                scotland_yard_game.GRID_SIZE * 2,  # distance to cop_1
                scotland_yard_game.GRID_SIZE,  # position x of cop_2
                scotland_yard_game.GRID_SIZE,  # position y or cop_2
                scotland_yard_game.GRID_SIZE * 2,  # distance to cop_2
                scotland_yard_game.GRID_SIZE,  # position x of cop_3
                scotland_yard_game.GRID_SIZE,  # position y or cop_3
                scotland_yard_game.GRID_SIZE * 2,  # distance to cop_3
            ]), dtype=np.float32),
            "cop_1": general_cop_observation_space,
            "cop_2": general_cop_observation_space,
            "cop_3": general_cop_observation_space
        })

    def step(self, action_dict: Dict[AgentID, int]) -> (
            Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]):
        # check if actions of all agents are valid

        if len(action_dict) == 0:
            exit(1, "No actions provided")
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
            self.game.turn_number += 1
            # Reveal mr x position if current turn is in REVEAL_POSITION_TURNS
            if self.game.get_current_turn_number() in scotland_yard_game.REVEAL_POSITION_TURNS:
                self.game.get_mr_x().last_known_position = self.game.get_mr_x().position

        # Return observations, rewards, terminates,truncateds, info
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
        rewards = self.game.get_rewards(invalid_actions_players)
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
