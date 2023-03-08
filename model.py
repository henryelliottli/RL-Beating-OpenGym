import gym
from gym import wrappers
import random
import numpy as np
from PIL import Image
from torchvision import models
import torchvision.transforms as T
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, num_stack_frames, image_size, actions):
        # image size is (h, w)
        super(DQN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_stack_frames, 32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 =nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size(conv2d_size(conv2d_size(image_size[1])))
        convh = conv2d_size(conv2d_size(conv2d_size(image_size[0])))
        input_size = convw * convh * 64
        # FC layers
        self.lin1 = nn.Linear(input_size, 128)
        self.output = nn.Linear(128, actions)
        
    
    def forward(self, x):
        # run through conv layer
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.relu(self.bn3(self.conv3(x2)))

        # run through linear layer
        x4 = self.relu(self.lin1(x3.view(x3.size(0), -1)))
        qscores = self.output(x4)
        return qscores
    
    def __call__(self, a):
        return self.forward(a)

    @classmethod
    def format_state(cls, state, device):
        state = state.__array__()
        state = state.transpose((3, 0, 1, 2))
        state = torch.autograd.Variable(torch.from_numpy(state).float())
        state = state.to(device)
        return state

class LinearDQN(nn.Module):
    def __init__(self, env):
        super().__init__()
        
        in_features = int(np.prod(env.observation_space.shape))
        hidden_units=128
        widen = int(in_features * 2)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.output = nn.Linear(hidden_units, env.action_space.n)
        
    def forward(self, x):
        x1 = self.relu(self.fc1(x))
        x2 = self.relu(self.fc2(x1))
        qscores = self.output(x2)
        return qscores

    def __call__(self, a):
        return self.forward(a)

    @classmethod
    def format_state(cls, state, device):
        state = state.__array__()
        state = torch.autograd.Variable(torch.from_numpy(state.flatten()).float().unsqueeze(0))
        state = state.to(device)
        return state


class DQNAgent:
    def __init__(self, train_model, target_model, env, device, lr, gamma, min_exp, max_exp, render_env, tau):
        # openai gym environment
        self.env = env
        self.memory = Memory(max_exp)
        self.min_exp = min_exp
        self.render_env = render_env
        self.tau = tau
        
        # setup model and train parameters
        self.gamma = gamma
        self.eps = 0.99
        self.batch_size = 32
        
        # model loss and optimizer
        self.device = device
        self.train_model = train_model
        self.target_model = target_model
        self.criterion = nn.HuberLoss(delta=2.0)
        self.optimizer = torch.optim.Adam(self.train_model.parameters(), lr=lr)
    
    def generate_action(self, state):
        # generate value to decide whether to use random action or best
        compare = np.random.rand(1)
        n_actions = self.env.action_space.n
        
        # choose a random action or best action
        if compare < self.eps:
            return torch.tensor([[np.random.randint(n_actions)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.train_model(state).max(1)[1].view(1,1)
        
    def update(self):
        # calculate Q and exp_Q from the batch
        if len(self.memory) < self.min_exp:
            return 0
        
        # grab batch from memory
        batch = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*batch))
        
        # torch.autograd.Variable(torch.from_numpy(state).float().unsqueeze(0))
        states = torch.cat(batch.state)
        states.to(self.device)
        next_states = torch.cat(batch.next_state)
        next_states.to(self.device)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        
        # Calc Q
        Q = self.train_model(states).gather(1, actions)
        
        # Calc Exp Q
        next_state_vals = self.target_model(next_states).max(1)[0].detach()
        exp_Q = (self.gamma * next_state_vals) + rewards
        
        #fixes a bug with numpy floats
        Q = Q.float()
        exp_Q = exp_Q.float().detach()

        # Calc Huber Loss
        loss = self.criterion(Q, exp_Q.unsqueeze(1).detach())
        
        # Train model 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
 
    def play(self):
        rewards = 0
        next_state = self.env.reset()
        next_state = self.train_model.format_state(next_state, self.device)
        done = False
        losses = []

        while not done:
            if self.render_env:
                self.env.render()
            # generate action to use in the environment and get data after using it
            action = self.generate_action(next_state)
            state = next_state
            next_state, reward, done, _ = self.env.step(action.item())
            next_state = self.train_model.format_state(next_state, self.device)
            reward = torch.tensor([reward], device=self.device)
            rewards += reward.item()
            
            # If the game finished punish it since we don't want the game to end
            if done:
                self.env.reset()
            
            # Train model after action
            self.memory.push(state, action, next_state, reward)
            loss = self.update()
            
            if self.tau:
                # update target network
                for target_param, local_param in zip(self.target_model.parameters(), self.train_model.parameters()):
                    target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
            
            # only is int if our experiences < min_exp
            if isinstance(loss, int):
                losses.append(loss)
            else:
                losses.append(loss.item())

        return rewards, np.mean(losses)
    
class Memory(object):
    def __init__(self, max_experiences):
        self.experiences = deque([],maxlen=max_experiences)
        
    def push(self, *args):
        # args - state, action, next_state, reward
        self.experiences.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.experiences, batch_size)

    def __len__(self):
        return len(self.experiences)