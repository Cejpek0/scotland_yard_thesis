import functools
import sys
from typing import List

from gymnasium.spaces import Box, Discrete
from numpy import random
from pettingzoo import AECEnv

import numpy as np
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AgentID, ObsType, ParallelEnv

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

    return env


class ScotlandYardPettingzoo_RAW(ParallelEnv):
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
            ]), dtype=np.float32),
        }

    def reset(self, **kwargs):
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

    def step(self, actions):
        print("---Step---")

        invalid_actions_players = []
        for player, action in actions.items():
            if player == 0:
                player = "mr_x"
            else:
                player = "cop_1"
            if action in range(0, 4):
                print(f"Action: {scotland_yard_game.Direction(action).name}")
            else:
                sys.stderr.write(f"Invalid action: {action}")
                exit(1)

            if self.game.get_current_turn_number() in scotland_yard_game.REVEAL_POSITION_TURNS:
                self.game.get_mr_x().last_known_position = self.game.get_mr_x().position

            # Execute actions
            if player == "mr_x":
                player = self.game.get_mr_x()
            else:
                player = self.game.get_cop_by_number(int(str.replace(player, "cop_", "")))

            if player is not None:
                invalid_actions_players.append(player)

        for player, action in actions:
            if player == 0:
                player = "mr_x"
            else:
                player = "cop_1"
            if action in range(0, 4):
                print(f"Action: {scotland_yard_game.Direction(action).name}")
            else:
                sys.stderr.write(f"Invalid action: {action}")
                exit(1)

            if self.game.get_current_turn_number() in scotland_yard_game.REVEAL_POSITION_TURNS:
                self.game.get_mr_x().last_known_position = self.game.get_mr_x().position

            # Execute actions
            if player == "mr_x":
                player = self.game.get_mr_x()
            else:
                player = self.game.get_cop_by_number(int(str.replace(player, "cop_", "")))

            if player is not None:
                if len(invalid_actions_players) == 0:
                    self.game.move_player(player, scotland_yard_game.Direction(action))
                    
            self.state[player] = action

        self.rewards = self.get_rewards(invalid_actions_players)
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

    def get_rewards(self, invalid_actions_players: List[str] = None):
        if invalid_actions_players is None:
            invalid_actions_players = []
        rewards = {}
        minimum_distance = 2
        # __ MR X __ #
        if "mr_x" in invalid_actions_players:
            rewards["mr_x"] = -100
        else:
            # Distance to cops
            distance_reward = 0
            
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
            if "cop_" + str(cop.number) in invalid_actions_players:
                rewards["cop_" + str(cop.number)] = -100
                continue
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
        game_over = self.game.is_game_over()
        returnDict = {}
        for agent_id in self.possible_agents:
            if game_over == scotland_yard_game.GameStatus.ONGOING:
                returnDict[agent_id] = False
            else:
                returnDict[agent_id] = True
        return returnDict
