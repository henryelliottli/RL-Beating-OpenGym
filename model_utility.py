import time
from model import DQN, LinearDQN, DQNAgent
from gym import wrappers
import numpy as np
import gc
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
gc.collect() 
torch.cuda.empty_cache()

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        torch.nn.init.normal_(m.bias.data)
        
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        torch.nn.init.normal_(m.bias.data)
           
def plot_average(avg, avg_title, output):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(avg)), avg)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(f"Average {avg_title}")
    ax.set_title(f"Average {avg_title} on Boxing")
    plt.savefig(output)
    return
    
def model_plays(env, model):
    model.eps = 0 # set epsilon to 0 so model plays entire time
    model.min_eps = 0
    env = wrappers.monitor(env, "videos", force=True)
    rewards, steps, done = 0, 0, False
    state = env.reset()
    while not done:
        action = model.generate_action(state)
        state, reward, done, _ = env.step(action)
        rewards += reward
    print(f"Rewards {rewards}")
    return
   
def create_cnn_networks(env, device, exp_config):
    # get environment parameters
    num_stacked_frames = env.observation_space.shape[0]
    image_size = env.observation_space.shape[1:3]
    action_size = env.action_space.n

    # setup models
    train_model = DQN(num_stacked_frames, image_size, action_size).to(device)
    train_model.apply(init_weights)
    target_model = DQN(num_stacked_frames, image_size, action_size).to(device)
    target_model.apply(init_weights)

    # create agent
    tau = exp_config['tau'] if exp_config['soft_update'] else None
    Agent = DQNAgent(train_model, target_model, env, device, exp_config['lr'],
                     exp_config['gamma'], exp_config['min_exp'],
                     exp_config['max_exp'], exp_config['render_env'], tau)

    return train_model, target_model, Agent

def create_ram_networks(env, device, exp_config):
    # setup models
    train_model = LinearDQN(env).to(device)
    train_model.apply(init_weights)
    target_model = LinearDQN(env).to(device)
    target_model.apply(init_weights)

    # create agent
    tau = exp_config['tau'] if exp_config['soft_update'] else None
    Agent = DQNAgent(train_model, target_model, env, device, exp_config['lr'],
                     exp_config['gamma'], exp_config['min_exp'],
                     exp_config['max_exp'], exp_config['render_env'], tau)

    return train_model, target_model, Agent

def create_models(env, device, use_ram, exp_config):
    if use_ram:
        print('create ram model')
        return create_ram_networks(env, device, exp_config)
    else:
        print('create cnn model')
        return create_cnn_networks(env, device, exp_config)
