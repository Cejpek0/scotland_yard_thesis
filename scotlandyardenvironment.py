import gymnasium as gym
import numpy
import numpy as np
from gymnasium import spaces
import scotland_yard_game

fps = 30


class ScotlandYardEnvironment(gym.Env):
    def __init__(self):
        self.done = False
        self.observation = numpy.array(None)
        self.game = scotland_yard_game.ScotlandYard()
        self.render_mode = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(1, 1), dtype=np.float32)

    def step(self, action):
        
        if 
        return self.observation, reward, done, info

    def reset(self, **kwargs):
        self.done = False
        self.game.reset()

        # Observations
        self.observation = self.get_observation()
        return self.observation

    def get_observation(self):
        observation = numpy.array(
            [self.game.get_mr_x().last_known_position])
        agents = self.game.get_agents()
        for agent in agents:
            numpy.append(observation, agent.position)
        return observation
