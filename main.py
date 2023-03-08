from make_env import make_env
from model_utility import plot_average, model_plays, create_models
import numpy as np
import torch
import gym
import sys
import os
import json
from pathlib import Path
import shutil
torch.cuda.empty_cache()


def main():
    # load in configs
    config_path = sys.argv[1]
    if os.path.isfile(config_path):
        with open(config_path) as json_file:
            config = json.load(json_file)
    else:
        raise Exception("file doesn't exist: ", config_path)

    # parameters for game
    game_config = config['game']
    game = game_config['game_name']
    use_ram = game_config['use_ram'] and game != 'mario'
    resume = game_config['resume_training']
    num_stacked_frames = game_config['num_stacked_frames']

    # parameters for experiment
    exp_config = config['experiment']
    experiment_dir = exp_config['experiment_dir'] + '/'
    copy_step = exp_config['copy_step']
    min_eps = exp_config['min_eps']
    epochs = exp_config['epochs']
    decay = exp_config['decay']
    soft_update=exp_config['soft_update']
    save_frequency = exp_config['save_frequency']

    # torch set up
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create environment
    env = make_env(game, use_ram=use_ram, num_stacked_frames=num_stacked_frames)

    # erase old experiment
    if not resume:
        dirpath = Path(experiment_dir)
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
    
    # load old experiment or create a new one
    if os.path.exists(experiment_dir) and resume:
        # fetch saved netork parameters
        state_dict = torch.load(os.path.join(experiment_dir, 'latest_model.pt'))

        # create model
        train_model, target_model, Agent = create_models(env, device, use_ram, exp_config)

        # load saved network parameters
        train_model.load_state_dict(state_dict['train_model'])
        target_model.load_state_dict(state_dict['target_model'])
        Agent.eps = state_dict['eps']
        optimizer = Agent.optimizer.load_state_dict(state_dict['optimizer'])
        
        # setup rewards, loss, and cur_epoch
        cur_epoch = np.load(experiment_dir+'cur_epoch.npy')
        avg_rewards = np.load(experiment_dir+'avg_rewards.npy')
        all_rewards = np.load(experiment_dir+'all_rewards.npy')
        avg_losses = np.load(experiment_dir+'avg_losses.npy')
        cur_epoch = state_dict['cur_epoch']
        
    else:
        # create experiment directory
        os.mkdir(experiment_dir)

        # create models
        train_model, target_model, Agent = create_models(env, device, use_ram, exp_config)

        # setup rewards and loss
        avg_rewards = np.zeros(epochs)
        all_rewards = np.zeros(epochs)
        avg_losses = np.zeros(epochs)
        cur_epoch = 0

    # train the model and output metrics
    print('start')
    for epoch in range(cur_epoch, epochs):
        reward, avg_loss = Agent.play()
        all_rewards[epoch] = reward
        avg_rewards[epoch] = np.mean(all_rewards[max(0, epoch - save_frequency):(epoch + 1)])
        avg_losses[epoch] = avg_loss
        
        if epoch % copy_step == 0 and not soft_update:
            Agent.target_model.load_state_dict(Agent.train_model.state_dict())
        
        if epoch % save_frequency == 1 or epoch == epochs-1:
            print(f"epoch {epoch}, avg reward: {avg_rewards[epoch]}, avg loss: {avg_loss}, eps: {Agent.eps}")
            
            # save model
            root_model_path = os.path.join(experiment_dir, 'latest_model.pt')
            train_model_dict = train_model.state_dict()
            target_model_dict = target_model.state_dict()
            state_dict = {'train_model': train_model_dict, 'target_model': target_model_dict,
                          'optimizer': Agent.optimizer.state_dict(), 'cur_epoch': epoch,
                          'eps': Agent.eps}
            torch.save(state_dict, root_model_path)
            
            # save rewards and losses
            np.save(experiment_dir+'cur_epoch.npy', epoch)
            np.save(experiment_dir+'avg_rewards.npy', avg_rewards)
            np.save(experiment_dir+'all_rewards.npy', all_rewards)
            np.save(experiment_dir+'avg_losses.npy', avg_losses)

        # decay epsilon
        Agent.eps = max(min_eps, Agent.eps * decay)

    
    # plot avg rewards, avg_loss, and create video
    plot_average(avg_rewards, 'Rewards', f'figures/average_rewards_{game}_img.png')
    plot_average(avg_losses, 'Loss', f'figures/average_loss_{game}_img.png')
    
    env.close()
    torch.cuda.empty_cache() 
    return


if __name__ == '__main__':
    main()
