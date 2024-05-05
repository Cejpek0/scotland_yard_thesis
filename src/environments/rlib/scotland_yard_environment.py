"""
This file contains the implementation of the Scotland Yard environment for the learning agents.
The environment is implemented as a MultiAgentEnv from the Ray RLlib library.
The environment is a turn-based game where the agents take turns to make a move.
Author: Michal Cejpek (xcejpe05@stud.fit.vutbr.cz)
"""
from typing import Dict, Any

import numpy as np
from gymnasium import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import AgentID

import src.game.scotland_yard_game_logic as scotland_yard_game


def policy_mapping_fn(agent_id, episode, worker):
    """Mapping function for the policy based on the agent_id"""
    return "mr_x_policy" if agent_id == "mr_x" else "cop_policy"


class ScotlandYardEnvironment(MultiAgentEnv):
    _agent_ids = ["mr_x", "cop_1", "cop_2", "cop_3"]

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.game = scotland_yard_game.ScotlandYardGameLogic()
        self.observations = None
        self._agent_ids = ["mr_x", "cop_1", "cop_2", "cop_3"]
        self.agents = self._agent_ids
        self.num_agents = len(self._agent_ids)

        # action space definition for each agent
        self.action_space = spaces.dict.Dict({
            "mr_x": spaces.Discrete(9),
            "cop_1": spaces.Discrete(9),
            "cop_2": spaces.Discrete(9),
            "cop_3": spaces.Discrete(9)
        })

        # observation space definition for each agent
        self.observation_space = spaces.dict.Dict({
            "mr_x": scotland_yard_game.mrx_observation_space,
            "cop_1": scotland_yard_game.general_cop_observation_space,
            "cop_2": scotland_yard_game.general_cop_observation_space,
            "cop_3": scotland_yard_game.general_cop_observation_space
        })

    def step(self, action_dict: Dict[AgentID, int]) -> (
            Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]):
        """
        Step function for the environment.
        Tries to perform the action of the current agent.
        If the action is invalid, heavily penalize the agent and do not proceed with the step.
        Returns observations for next agent, rewards of current agent,
        :param action_dict: Dictionary of actions for each agent
        :return: Observations, rewards, terminations, truncated, info
        """

        assert len(action_dict) > 0
        current_agent = self.game.get_current_player()
        action = action_dict[current_agent.name]
        rewards = {agent: 0 for agent in self._agent_ids}
        direction = scotland_yard_game.Direction(action)

        # check if actions of all agents are valid
        invalid_move = not self.game.is_valid_move(current_agent, direction)

        # If action is invalid, do not proceed with the step. Do not update the game, and heavily penalize agents
        # that made invalid actions
        if invalid_move:
            observations = {current_agent.name: self.get_observations()[current_agent.name]}
            infos = {current_agent.name: {}}
            temp_rewards = self.get_rewards(current_agent.name)
            rewards[current_agent.name] = temp_rewards[current_agent.name]
            return observations, rewards, self.getTerminations(), {
                "__all__": False, "mr_x": False, "cop_1": False, "cop_2": False, "cop_3": False, }, infos

        self.game.play_turn(direction)

        # get next player
        next_player = self.game.get_player_by_number((current_agent.number + 1) % self.num_agents)
        assert next_player is not None
        infos = {}

        # Calculate rewards
        turn_reward = self.get_rewards()
        if self.game.get_game_status() == scotland_yard_game.GameStatus.ONGOING:
            rewards[current_agent.name] = turn_reward[current_agent.name]
        else:
            rewards = turn_reward

        # Check if the game is over
        terminates = self.getTerminations()
        # Get updated observations
        observations = {next_player.name: self.get_observations()[next_player.name]}

        # Return observations, rewards, terminates,truncated, info
        return observations, rewards, terminates, {"__all__": False, "mr_x": False, "cop_1": False, "cop_2": False,
                                                   "cop_3": False, }, infos

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to the initial state.
        :return: Observations, infos
        """
        self.game.reset()
        current_player = self.game.get_current_player()
        # Observations
        observations = {current_player.name: self.get_observations()[current_player.name]}

        return observations, {agent: None for agent in self._agent_ids}

    def get_rewards(self, invalid_actions_player: str | None = None) -> {str: float | None}:
        """
        Calculate rewards for all agents
        win_rewards: Rewards for winning the game: 50 for direct win. 30 for indirect win. -20 for losing
        distance_rewards: Rewards for being close to the target: capped at 10
        inactivity_penalty: Penalty for being inactive: starts after 5 turns: -0.5 per turn including the 5 turns
        inside_rewards: Rewards for being inside the area of interest: 12
        invalid_actions_player: Penalize the agent that made an invalid action: -20
        :param invalid_actions_player: Player that made an invalid action
        :return: Dictionary of rewards for all agents: {agent_id: reward}
        """
        if invalid_actions_player is not None:
            return {invalid_actions_player: -20}
        minimum_distance = 4
        rewards = {agent: 0.0 for agent in self._agent_ids}
        win_rewards = {agent: 0.0 for agent in self._agent_ids}
        inactivity_penalty = {agent: 0.0 for agent in self._agent_ids}
        distance_rewards = {agent: 0.0 for agent in self._agent_ids}
        inside_rewards = {agent: 0.0 for agent in self._agent_ids}

        # Win rewards #
        if self.game.get_game_status() == scotland_yard_game.GameStatus.COPS_WON:
            for cop in self.game.get_cops():
                distance_to_mr_x = cop.get_distance_to(self.game.get_mr_x().position)
                if distance_to_mr_x == 0:
                    win_rewards[cop.name] = 50
                else:
                    win_rewards[cop.name] = 30
            win_rewards["mr_x"] = -50
        elif self.game.get_game_status() == scotland_yard_game.GameStatus.MR_X_WON:
            win_rewards["mr_x"] = 50
            win_rewards["cop_1"] = -20
            win_rewards["cop_2"] = -20
            win_rewards["cop_3"] = -20

        # Inactivity penalty #
        for agent in self._agent_ids:
            if self.game.agents_is_in_previous_location_count[agent] > 5:
                inactivity_penalty[agent] = self.game.agents_is_in_previous_location_count[agent] * (-0.5)
            else:
                inactivity_penalty[agent] = 0

        # __ MR X __ #
        # Distance to cops capped at 20 and - 20
        closest_cop_distance = scotland_yard_game.MAX_DISTANCE
        for cop in self.game.get_cops():
            distance = self.game.get_mr_x().get_distance_to(cop.position)
            if distance < closest_cop_distance:
                closest_cop_distance = distance
        distance_rewards["mr_x"] = round(
            (((closest_cop_distance - minimum_distance) / (scotland_yard_game.MAX_DISTANCE - minimum_distance)) * 20),
            10)

        # __ COPS __ #
        possible_mr_x_positions = self.game.get_possible_mr_x_positions()

        for cop in self.game.get_cops():
            # If the cop is inside the area of interest
            if cop.position in possible_mr_x_positions:
                # Being inside the area of interest
                inside_rewards[cop.name] = 12
            else:
                # Closer to the area of interest, the more reward is gained
                # Maximum reward is 10, so being inside location of interest is more beneficial
                closest_position = self.game.get_closest_position(
                    cop.position,
                    possible_mr_x_positions
                )
                distance_to_closest_position = cop.get_distance_to(closest_position)

                distance_rewards[cop.name] = (float(
                    minimum_distance - distance_to_closest_position) / scotland_yard_game.MAX_DISTANCE) * 10

        # Sum all rewards
        for agent in self._agent_ids:
            rewards[agent] += (
                    distance_rewards[agent]
                    + win_rewards[agent]
                    + inactivity_penalty[agent]
                    + inside_rewards[agent]
            )
        return rewards

    def get_observations(self):
        """
        Get observations for all agents
        :return: Dictionary of observations for all agents: {agent_id: observation}
        """
        observations = self.game.get_observations()
        return observations

    def get_empty_observation(self):
        """
        Get empty observation for all agents
        :return: Dictionary of empty observations for all agents: {agent_id: observation}
        """
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
        """
        Get terminations for all agents.
        Terminations depend on the game status.
        :return: Dictionary of terminations for all agents: {agent_id: termination}
        """
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
