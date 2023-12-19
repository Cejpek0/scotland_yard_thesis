from typing import Dict, Any

import numpy as np
from gymnasium import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import AgentID

import scotland_yard_game

_agent_ids = ["mr_x", "cop"]


class FakeEnv(MultiAgentEnv):
    _agent_ids = ["mr_x", "cop"]

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.observations = None
        self._agent_ids = ["mr_x", "cop_1"]
        self.num_agents = len(self._agent_ids)

        self.action_space = {"mr_x": spaces.Discrete(4), "cop_1": spaces.Discrete(4)}

        self.observation_space = {
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
        }

    def step(self, action_dict: Dict[AgentID, int]) -> (
            Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]):

        return self.get_empty_observation(), {"mr_x": 0, "cop_1": 0}, {"__all__": False, "mr_x": False, "cop_1": False}, {"__all__": False, "mr_x": False, "cop_1": False}, {"mr_x": {},"cop_1": {}}

    def reset(self, *, seed=None, options=None):
        return self.get_empty_observation(), {"mr_x": 0, "cop_1": 0}

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
