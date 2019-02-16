import gym
from gym import error, spaces, utils
from gym.utils import seeding

class igcEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    print("Init")
    
  def step(self, action):
    print("step")
    
  def reset(self):
    print("reset")
      
  def render(self, mode='human', close=False):
    print("render")
