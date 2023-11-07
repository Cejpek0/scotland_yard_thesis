import numpy
from gym import spaces
from ray.rllib import MultiAgentEnv

import scotland_yard_game

class ScotlandYardEnvironment(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.game = scotland_yard_game.ScotlandYard()
        self.num_agents = self.game.players.count()
        self.observations = None

        # Define your environment state and other components here
        self.action_space = {"mr_x": spaces.Discrete(4)}
        for cop in self.game.get_cops():
            self.action_space["cop" + cop.number] = spaces.Discrete(4)
        mr_x_number_of_observations = 7 # current turn, max turns, next_reveal, position (x,y), last known position (x,y),
        for cop in range(self.game.get_cops().count()):
            mr_x_number_of_observations += 3 # 1 distance, 2 position (x,y)
        cop_number_of_observations = 7 # current turn, max turns, next_reveal, position, last known position of mr_x (x,y)
        for cop in range(self.game.get_cops().count()-1):
            cop_number_of_observations += 2
        self.observation_space = {"mr_x": spaces.Box(low=-100, high=100, shape=(mr_x_number_of_observations,))}
        for cop in self.game.get_cops():
            self.observation_space["cop_" + cop.number] = spaces.Box(low=-100,
                                                                     high=100,
                                                                     shape=(cop_number_of_observations,))

    def step(self, action):
        
        return self.observations, reward, done, info

    def reset(self, **kwargs):
        self.game.reset()

        # Observations
        self.observations = self.get_observation()
        return self.observations

    def get_observation(self):
        observations = {}
        arr = []
        # mr_x
        arr.append(self.game.get_current_turn_number())
        arr.append(self.game.get_max_turns())
        arr.append(self.game.next_reveal_turn_number())
        arr.append(self.game.get_mr_x().position[0])
        arr.append(self.game.get_mr_x().position[1])
        for cop in self.game.get_cops():
            arr.append(self.game.get_mr_x().get_distance_to(cop.position))
            arr.append(cop.position[0])
            arr.append(cop.position[1])
        observations["mr_x"] = arr
        arr = []
        return observations