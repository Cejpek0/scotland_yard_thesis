import numpy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor

import os

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

model = PPO('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log=log_dir)