import functools
import sys

from gymnasium.spaces import Box, Discrete
from numpy import random
from pettingzoo import AECEnv

import numpy as np
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AgentID, ObsType

import scotland_yard_game
from gymnasium import spaces
from pettingzoo.utils import agent_selector, wrappers


def ScotlandYardPettingzoo(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = ScotlandYardPettingzoo_RAW(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide variety of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class ScotlandYardPettingzoo_RAW(AECEnv):
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "scotland_yard_pettingzoo",
        "is_parallelizable": True,
    }

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        super().__init__()
        scotland_yard_game.NUMBER_OF_COPS = 1
        self.game = scotland_yard_game.ScotlandYard()
        self.possible_agents = ["mr_x", "cop_1"]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.action_spaces = {"mr_x": spaces.Discrete(4), "cop_1": spaces.Discrete(4)}

        spaces.Tuple((spaces.Discrete(4), spaces.Discrete(4)))
        self.observation_spaces = {
            "mr_x": spaces.Dict({
                "observation": spaces.Box(low=np.array([
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
                "action_mask": spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8)}),
            "cop_1": spaces.Dict({
                "observation": spaces.Box(low=np.array([
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
                ]), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8)})
        }

    def reset(self, seed=None, options=None):
        print("Reset")
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
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = self.get_empty_observation()
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        print("---Step---")
        print(f"Agent: {self.agent_selection}")
        
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        if action in range(0,4):
            print(f"Action: {scotland_yard_game.Direction(action).name}")
        else:
            sys.stderr.write(f"Invalid action: {action}")
            exit(1)
        
        if self.game.get_current_turn_number() in scotland_yard_game.REVEAL_POSITION_TURNS:
            self.game.get_mr_x().last_known_position = self.game.get_mr_x().position

        # Execute actions
        if self.agent_selection == "mr_x":
            player = self.game.get_mr_x()
        else:
            player = self.game.get_cop_by_number(int(str.replace(self.agent_selection, "cop_", "")))
        if player is not None:
            self.game.move_player(player, scotland_yard_game.Direction(action))

        self._cumulative_rewards[self.agent_selection] = 0

        self.state[self.agent_selection] = action



        self.rewards = self.get_rewards()
        self.terminations = self.getTerminations()
        self.observations = self.get_observations()
        self.truncations = {agent: False for agent in self.agents}

        # Update game state            
        if self._agent_selector.is_last():
            self.num_moves += 1
            self.game.turn_number += 1
        else:
            if self.agent_selection == "mr_x":
                self.state["cop_1"] = None
            else:
                self.state["mr_x"] = None
            # no rewards are allocated until both players give an action
            self._clear_rewards()
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.render_mode == "ansii":
            self.render()

    def render(self):
        self.game.text_display()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def close(self):
        pass

    def observe(self, agent: AgentID) -> ObsType | None:
        return self.observations[agent]

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
                distance = self.game.get_mr_x().get_distance_to(cop)
                if distance == 0:
                    distance_reward += 1
                distance_reward += round((distance - minimum_distance) * 0.1, 10)
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

        mr_x_action_mask = self.game.get_players_valid_moves_mask(mr_x)
        cop_action_mask = self.game.get_players_valid_moves_mask(cop)

        observations = {
            "mr_x": {"observation": obs_list_mrx, "action_mask": np.array(mr_x_action_mask).astype(np.int8)},
            "cop_1": {"observation": obs_list_cop, "action_mask": np.array(cop_action_mask).astype(np.int8)}
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
        mr_x_action_mask = self.game.get_players_valid_moves_mask(self.game.get_mr_x())
        cop_action_mask = self.game.get_players_valid_moves_mask(self.game.get_cops()[0])
        return {
            "mr_x": {"observation": mrx_obs, "action_mask": np.array(mr_x_action_mask).astype(np.int8)},
            "cop_1": {"observation": cop_obs, "action_mask": np.array(cop_action_mask).astype(np.int8)}
        }

    def getTerminations(self):
        game_over = self.game.is_game_over()
        returnDict = {}
        for agent_id in self.possible_agents:
            if game_over == scotland_yard_game.GameStatus.ONGOING:
                returnDict[agent_id] = False
            else:
                returnDict[agent_id] = True
        return returnDict
    
    def action_mask(self):
        return self.observations[self.agent_selection]["action_mask"]