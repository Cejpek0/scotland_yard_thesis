from typing import Dict, List, Any

import numpy as np
from gymnasium import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import AgentID
import random


class FakeEnv(MultiAgentEnv):
    _agent_ids = ["mr_x", "cop_1", "cop_2", "cop_3"]

    def __init__(self, config):
        self.next_agent_index = 1
        self.next_agent = "cop_1"
        super().__init__()
        self.config = config
        self.observations = None
        self._agent_ids = ["mr_x", "cop_1", "cop_2", "cop_3"]

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
        ]), high=np.array([
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
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
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
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
        
        observations = self.get_empty_observation()
        observation = {agent_id: observations[agent_id]}
        infos = {agent_id: {}}
        

        return observation, {"mr_x": 0, "cop_1": 0, "cop_2": 0, "cop_3": 0}, {
            "__all__": False, "mr_x": False, "cop_1": False, "cop_2": False, "cop_3": False, }, {
            "__all__": False, "mr_x": False, "cop_1": False, "cop_2": False, "cop_3": False, }, infos

    def reset(self, *, seed=None, options=None):
        observations = self.get_empty_observation()
        self.next_agent_index = 1
        self.next_agent = "cop_1"

        return observations, {"mr_x": 0, "cop_1": 0, "cop_2": 0, "cop_3": 0}

    def get_empty_observation(self):
        mrx_obs = np.array([0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        mrx_obs = mrx_obs.astype(np.float32)
        cop_obs = np.array([0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0])
        cop_obs = cop_obs.astype(np.float32)
        return {
            "mr_x": mrx_obs
        }
