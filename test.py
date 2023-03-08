from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack, GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from make_env import make_env
from matplotlib import pyplot as plt
import os
import sys
import time
from stable_baselines3 import PPO
#reloaded base callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
import torch
from model import *
from model_utility import init_weights, plot_average, model_plays


game = sys.argv[1]
if game == 'Mario':
    env = gym_super_mario_bros.make('SuperMarioBros-v3')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    #grayscale
    env = GrayScaleObservation(env, keep_dim=True)

    #wrap it inside dummyENV
    # env = DummyVecEnv([lambda: env])

    # #Stack the Frames
    # env = VecFrameStack(env, 4, channels_order = 'last')

else:
    env = make_env(game)
    
state = env.reset()
print(state.shape)

# Load model
state_dict = torch.load(f'{game}_final_experiment/latest_model.pt')
# setup models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
action_size = env.action_space.n
image_size = env.observation_space.shape[:2]

train_model = DQN(image_size, action_size).to(device)
train_model.apply(init_weights)
train_model.load_state_dict(state_dict['train_model'])

target_model = DQN(image_size, action_size).to(device)
target_model.apply(init_weights)
target_model.load_state_dict(state_dict['target_model'])
print(type(state_dict['train_model']))
# optimizer = torch.optim.Adam(train_model.parameters(), lr=1e-2)
# optimizer = optimizer.load_state_dict(state_dict['optimizer'])

# Start the game 
state = env.reset()
state = state.transpose((2, 0, 1))
state = torch.autograd.Variable(torch.from_numpy(state).float().unsqueeze(0))
state = state.to(device)
Agent = DQNAgent(train_model, target_model, env, device)

# Loop through the game

for epoch in range(5000):
    Agent.eps = 0
    action = Agent.generate_action(state)
    new_state, reward, done, info = env.step(action.item())
    env.render()
    new_state = new_state.transpose((2, 0, 1))
    new_state = torch.autograd.Variable(torch.from_numpy(new_state).float().unsqueeze(0))
    state = new_state.to(device)
    time.sleep(0.1)
    if done:
        break
    

# while True: 
#     action, _ = model.predict(state)
#     state, reward, done, info = env.step(action)
#     env.render()