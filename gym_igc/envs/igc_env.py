import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matlab.engine
import numpy as np

##############################################
# igc environment to work within OpenAI Gyms framework
# Simulator class that will handle:
#      (1) Simulator Initialization
#      (2) Simulator calls
#          - input insulin value into simulator
#          - Recieve glucose values from simulator
#          - Calculate reward for deep reinforcement learning model
#      (3) Simulator Cleanup
#
#         The Patient simulator is written in matlab
#         It is leveraged here by making calls to the simulator functions
#         through a matlab-python interface provided by mathworks
#############################################
class igcEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  ########################
  # Constructor
  #       Initialize matlab engine
  #       selmode is optional parameter
  #           0 randomly samples initial patient state and parameters
  #           1 chooses hardcoded patient state and parameter vector
  #       Saves initialization parameters
  #       x0 - inital state vector
  #       p - patient parameter vector
  #       t - time 
  #       x - current patient state
  #       curBG - current observable blood glucose level
  #       inv1 - 
  #       inv2 -
  #       mode -
  #######################
  def __init__(self):
    
    # DDPG requires symmetric action spaces... will have to rescale accordingly    
    self.action_low = -25 # cannot give negative insulin
    self.action_high = 25 # I will have to ask someone about this

    self.observation_low = 0 # cannot have negative blood glucose
    self.observation_high = 500 # I am unsure about this...

    # two dimensional action space
    self.action_space = spaces.Box(low=self.action_low,high=self.action_high,shape=(2,),dtype=np.float64)
    
    self.observation_space = spaces.Box(low=self.observation_low,high=self.observation_high,shape=(1,),dtype=np.float32)
    
    self.eng = matlab.engine.start_matlab()
    data = self.eng.init_simulator(1);
    self.p = data['p']
    self.x = data['x']
    
    self.curBG = data['curBG']


  #############################################################################
  # Interact with the simulator. "Step the environment"
  # Input
  #      action - input from controller
  # From current episode
  #      x - current state
  #      p - patient parameters
  # Return
  #      curBG - current observable state for agent
  #      data['reward'] - reward for previous action from agent
  #      done - True indicates episode terminated (i.e. surgery has finished)
  #             returning True will force environment reset
  #           - false will continue current episode training loop
  #      {} - I can fit diagnostic info here in the framework
  ###########################################################################
  def step(self, action):
    
    self.episode_counter = self.episode_counter + 1;

    # add m(t) to state before going into environment
    self.x[0][5] = self.x[0][5] + action[1]

    # rescale action to appropriate range 
    action[0] = action[0] + 25

    data = self.eng.simulation_step(action[0].item(), self.x, self.p)
    self.x = data['x']
    self.curBG = data['curBG']

    if self.episode_counter == 10:
      done = True
    else:
      done = False
    
    return self.curBG, data['reward'], done, {}

  #############################
  # Initalize a new patient
  #      selmode is optional parameter
  #           0 randomly samples initial patient state and parameters
  #           1 chooses hardcoded patient state and parameter vector
  ############################  
  def reset(self):
    
    self.episode_counter = 0
    
    data = self.eng.init_simulator(1);
    self.p = data['p']
    self.x = data['x']
    self.curBG = data['curBG']
    
    return self.curBG

  # unnecessary for this environment
  def render(self, mode='human', close=False):
    print("render")
