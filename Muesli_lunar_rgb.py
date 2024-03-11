import math
import time
import os
import argparse

import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageDraw as ImageDraw
import cv2
import numpy as np

import nni

#import torchsummary


params = {
    ## Params controlled by this file
    'game_name': 'LunarLander-v2', # gym env name
    'actor_max_epi_len': 1000, # max step of gym env
    'success_threshold': 200, # arbitrary success threshold of game score

    'RGB_Wrapper': True, # change vector based state to RGB
    'norm_factor': 255.0, # normalize RGB by /255.0
    'resizing_state': True, # overwrite the H,W related params to resize
    'resize_height': 72, # image resize H
    'resize_width': 96, # image resize W
    
    'draw_wrapped_state': False, # draw image which agent see actually
    'draw_image': True, # draw full resolution RGB episode
    'draw_per_episode': 500, # drawing slows down the code, adjust frequency by this
    'negative_reward': False, # experimental feature, making last zero reward to negative reward
    'negative_reward_val': -100.0, # negative reward value
    'stack_action_plane': False, #True, #False, # stack action information plane to RGB state 
    
    'beta_var': 0.99, # related to advantage normalization
    'eps_var': 1e-12, # related to advantage normalization
    'hs_resolution': 512, # resolution of hidden state
    'mlp_width': 128,  # mlp network width
    'support_size': 30, # support_size of categorical representation
    'eps': 0.001, # categorical representation related
    'discount': 0.997, # discount rate
    'start_lr': 0.0003, # learning rate
    'expriment_length': 20000, #4000, # num of repetitions of self-play&update  
    'replay_proportion': 75, # proportion of the replay inside minibatch
    'unroll_step' : 4, # unroll step
    'adv_clip_val': 1, # adv normalize clip value

    'total_policy_loss_weight': 1, # total policy loss weight
    'value_loss_weight': 0.25, # multiplier for value loss
    'reward_loss_weight': 1, # multiplier for reward loss

    ## HPO params controlled by config.yaml
    'use_last_fc': True, # related to represenation 
    'use_fixed_random_seed': True, # use fixed random seed on the Gym enviornment
    'random_seed': 42, # random seed
    'use_proj': True, # use projection with mlp in the networks, True is recommended
    'second_term_weight': 1, # regularizer term weight
    'mixed_prior': True, # using mixed pi_prior
    'stacking_frame': 4, # stacking previous states
    'mb_dim': 128, # dimension of minibatch 
    'iteration': 20, # num of iteration 
    'alpha_target': 0.01, # target network(prior parameters) moving average update ratio

    ## Params will be assigned by the code
        # params['input_height']
        # params['input_width']
        # params['input_channels']
        # params['action_space']
}

optimized_params = nni.get_next_parameter()
params.update(optimized_params)

## For linear lr decay
## https://github.com/cmpark0126/pytorch-polynomial-lr-decay
from torch_poly_lr_decay import PolynomialLRDecay


############################ Models ############################

def mlp_proj(input_size, output_size):
    return torch.nn.Sequential(
            nn.Linear(input_size, params['mlp_width']),
            nn.ReLU(),
            nn.Linear(params['mlp_width'], output_size),
            nn.ReLU(),
        )
    

class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )

class Representation(nn.Module): 
    def __init__(self, input_channels, hidden_size, mlp_width):   
        super().__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, (3, 3), stride=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResNet(
                torch.nn.Sequential(
                    torch.nn.ReLU(),
                    conv3x3(16, 16, 1),
                    torch.nn.ReLU(),
                    conv3x3(16, 16, 1),
                )
            ),
            ResNet(
                torch.nn.Sequential(
                    torch.nn.ReLU(),
                    conv3x3(16, 16, 1),
                    torch.nn.ReLU(),
                    conv3x3(16, 16, 1),
                )
            ),
            nn.Conv2d(16, 32, (3, 3), stride=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResNet(
                torch.nn.Sequential(
                    torch.nn.ReLU(),
                    conv3x3(32, 32, 1),
                    torch.nn.ReLU(),
                    conv3x3(32, 32, 1),
                )
            ),
            ResNet(
                torch.nn.Sequential(
                    torch.nn.ReLU(),
                    conv3x3(32, 32, 1),
                    torch.nn.ReLU(),
                    conv3x3(32, 32, 1),
                )
            ),
            nn.Conv2d(32, 16, (3, 3), stride=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResNet(
                torch.nn.Sequential(
                    torch.nn.ReLU(),
                    conv3x3(16, 16, 1),
                    torch.nn.ReLU(),
                    conv3x3(16, 16, 1),
                )
            ),
            ResNet(
                torch.nn.Sequential(
                    torch.nn.ReLU(),
                    conv3x3(16, 16, 1),
                    torch.nn.ReLU(),
                    conv3x3(16, 16, 1),
                )
            ),
            nn.ReLU(),
        )        
        if params['use_last_fc']:
            self.fc_0 = nn.Linear(1408, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size) 
        else:
            self.lstm = nn.LSTM(1408, hidden_size) 

        self.proj_0 = mlp_proj(hidden_size, hidden_size)
        self.proj_1 = mlp_proj(hidden_size, hidden_size)         

    def forward(self, x):
        x = x.div(params['norm_factor'])
        x = self.image_conv(x)
        x = torch.flatten(x, start_dim=1)
        if params['use_last_fc']:
            x = F.relu(self.fc_0(x))
        x, hc = self.lstm(x.unsqueeze(0))
        x = x.squeeze(0)
        pre_p, pre_v = x, x
        if params['use_proj']:
            pre_p, pre_v = self.proj_0(pre_p), self.proj_1(pre_v)
        pre_p, pre_v = min_max_norm(pre_p), min_max_norm(pre_v)
        
        return pre_p, pre_v, hc



class Dynamics(nn.Module): 
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size) 
        self.proj_0 = mlp_proj(hidden_size, hidden_size)
        self.proj_1 = mlp_proj(hidden_size, hidden_size)
        self.proj_2 = mlp_proj(hidden_size, hidden_size)

        self.one_hot_act = torch.cat((torch.nn.functional.one_hot(torch.arange(0, params['action_space']) % params['action_space'], num_classes=params['action_space']), torch.zeros(params['action_space']).unsqueeze(0)), dim=0).to(device)  

    def forward(self, action, hc):
        action = action.squeeze(-1)
        action = torch.stack([
            self.one_hot_act[idx.to(torch.int64)] for idx in action
        ], dim=0)     
        output, _ = self.lstm(action, hc)      
        pre_p, pre_v, pre_r = output, output, output
        if params['use_proj']:
            pre_p, pre_v, pre_r = self.proj_0(pre_p), self.proj_1(pre_v), self.proj_2(pre_r)
        pre_p, pre_v, pre_r = min_max_norm(pre_p), min_max_norm(pre_v), min_max_norm(pre_r)
        
        return pre_p, pre_v, pre_r



class Policy(nn.Module): 
    def __init__(self, input_dim, width):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(input_dim,width),
            nn.ReLU(),
            nn.Linear(width,width),
            nn.ReLU(),
            nn.Linear(width, params['action_space'])           
        ) 

    def forward(self, x):
        x = self.policy_head(x)
        x = torch.nn.functional.softmax(x, dim=-1) 
        return x

class Value(nn.Module): 
    def __init__(self, input_dim, width):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(input_dim,width),
            nn.ReLU(),
            nn.Linear(width,width),
            nn.ReLU(),
            nn.Linear(width,support_size*2+1)           
        ) 

    def forward(self, x):
        x = self.value_head(x)
        x = torch.nn.functional.softmax(x, dim=-1) 
        return x

class Reward(nn.Module): 
    def __init__(self, input_dim, width):
        super().__init__()
        self.reward_head = nn.Sequential(
            nn.Linear(input_dim,width),
            nn.ReLU(),
            nn.Linear(width,width),
            nn.ReLU(),
            nn.Linear(width,support_size*2+1)           
        ) 

    def forward(self, x):
        x = self.reward_head(x)
        x = torch.nn.functional.softmax(x, dim=-1) 
        return x


############################ Utils ############################


def min_max_norm(x):
    x_min = x.min(-1, keepdim=True)[0]
    x_max = x.max(-1, keepdim=True)[0]
    x_normalized = (x - x_min) / (x_max - x_min)
    return x_normalized

support_size = params['support_size']
eps = params['eps']

def to_scalar(x):
    probabilities = x
    support = (torch.tensor([x for x in range(-support_size, support_size + 1)]).expand(probabilities.shape).float().to(device))
    x = torch.sum(support * probabilities, dim=1, keepdim=True)
    scalar = torch.sign(x) * (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps))** 2 - 1)
    return scalar

def to_scalar_with_soft(x): ## test purpose
    x = torch.softmax(x, dim=-1)
    probabilities = x 
    support = (torch.tensor([x for x in range(-support_size, support_size + 1)]).expand(probabilities.shape).float().to(device))
    x = torch.sum(support * probabilities, dim=1, keepdim=True)
    scalar = torch.sign(x) * (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps))** 2 - 1)
    return scalar

def to_cr(x):
    x = x.squeeze(-1).unsqueeze(0)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x
    x = torch.clip(x, -support_size, support_size)
    floor = x.floor()
    under = x - floor
    floor_prob = (1 - under)
    under_prob = under
    floor_index = floor + support_size
    under_index = floor + support_size + 1
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).type(torch.float32).to(device)
    logits.scatter_(2, floor_index.long().unsqueeze(-1), floor_prob.unsqueeze(-1))
    under_prob = under_prob.masked_fill_(2 * support_size < under_index, 0.0)
    under_index = under_index.masked_fill_(2 * support_size < under_index, 0.0)
    logits.scatter_(2, under_index.long().unsqueeze(-1), under_prob.unsqueeze(-1))
    return logits.squeeze(0)


class debug_time:
    """execution time checker
    
    with debug_time("My Custom Block", index):
        # code block to measure
    """
    def __init__(self, name="", global_i=0):
        self.name = name
        self.global_i = global_i

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        writer.add_scalar(f"Time/{self.name}", duration, global_i)


def draw_epi_act_rew(frame, episode_num, action, reward, score):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    text_color = (255,255,255)   
    drawer.text((im.size[0]/20,im.size[1]/18), f'Epi: {episode_num}   Act: {action}   Rew: {reward:.3f}\nScore: {score:.3f}', fill=text_color)
    im = np.array(im)
    return im

def draw_pi(frame, probabilities=None):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    text_color = (0,0,0)        
    drawer.text((im.size[0]/20,im.size[1]*16/18), f"Pi: {np.array2string(probabilities, formatter={'float': lambda x: f'{x:.2f}'}, separator=', ')}", fill=text_color)
    im = np.array(im)
    return im

def draw_val(frame, val=None):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    text_color = (255,255,255)         
    drawer.text((im.size[0]*14/20,im.size[1]/18), f'predicted_Val: {val:.7f}', fill=text_color)
    im = np.array(im)
    return im



############################ Main ############################

##Target network
class Target(nn.Module):
    """Target Network
    
    Target network is used to approximate v_pi_prior, q_pi_prior, pi_prior.
    It contains older network parameters. (exponential moving average update)
    """
    def __init__(self):
        super().__init__()
        self.representation_network = Representation(params['stacking_frame']*params['input_channels'], params['hs_resolution'], params['mlp_width']) 
        self.dynamics_network = Dynamics(params['action_space'], params['hs_resolution'])
        self.policy_network = Policy(params['hs_resolution'], params['mlp_width'])
        self.value_network = Value(params['hs_resolution'], params['mlp_width'])
        self.reward_network = Reward(params['hs_resolution'], params['mlp_width'])

        self.to(device)


##Muesli agent
class Agent(nn.Module):
    """Agent Class"""
    def __init__(self):
        super().__init__()

        self.env = gym.make(params['game_name'], render_mode="rgb_array")
        if params['RGB_Wrapper']:
            self.env = gym.wrappers.PixelObservationWrapper(self.env)
        params['input_height'], params['input_width'], params['input_channels'] = self.env.observation_space['pixels'].shape    
        if params['resizing_state']:
            params['input_height'], params['input_width'] = params['resize_height'], params['resize_width']

        if params['stack_action_plane']:
            params['input_channels'] += 1
        params['action_space'] = self.env.action_space.n
        
        self.representation_network = Representation(params['stacking_frame']*params['input_channels'], params['hs_resolution'], params['mlp_width']) 
        self.dynamics_network = Dynamics(params['action_space'], params['hs_resolution'])
        self.policy_network = Policy(params['hs_resolution'], params['mlp_width'])
        self.value_network = Value(params['hs_resolution'], params['mlp_width'])
        self.reward_network = Reward(params['hs_resolution'], params['mlp_width'])
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=params['start_lr'], weight_decay=0)
        self.scheduler = PolynomialLRDecay(self.optimizer, max_decay_steps=params['expriment_length'], end_learning_rate=0.0000)   
        self.to(device)

        self.state_replay = []
        self.action_replay = []
        self.P_replay = []
        self.r_replay = []   

        self.var = 0
        self.beta_product = 1.0

        self.var_m = [0 for _ in range(params['unroll_step']+1)] 
        self.beta_product_m = [1.0 for _ in range(params['unroll_step']+1)] 


    def self_play_mu(self, target, max_timestep=params['actor_max_epi_len']):  
        
        self.state_traj = []
        self.action_traj = []
        self.P_traj = []
        self.r_traj = []      

        game_score = 0
        action = 0
        r = 0
        last_frame = 1000

        if params['use_fixed_random_seed']:
            state = self.env.reset(seed=params['random_seed'])
        else:
            state = self.env.reset()
        
        state_image = cv2.resize(state[0]['pixels'], (params['resize_width'], params['resize_height']), interpolation=cv2.INTER_AREA).transpose(2,0,1)
        
        if params['stack_action_plane']:
            previous_action_plane = np.full((1, params['input_height'], params['input_width']), action/params['action_space']*params['norm_factor'])
            state = np.vstack((state_image, previous_action_plane))
        else:
            state = state_image
        
        for i in range(max_timestep):
            if params['draw_image'] and global_i % params['draw_per_episode'] == 0:
                img = draw_epi_act_rew(self.env.render(), episode_num=i, action=action, reward=r, score=game_score)
            
            if i == 0:
                for _ in range(params['stacking_frame']):
                    self.state_traj.append(state)      
                stacked_state = np.tile(state, (params['stacking_frame'], 1, 1))                 
            else:
                self.state_traj.append(state)
                stacked_state = np.roll(stacked_state, shift=-1*params['input_channels'] ,axis=0)
                stacked_state[-1*params['input_channels']:]=state                
            
            with torch.no_grad():
                output_p , output_v, _ = target.representation_network(torch.from_numpy(stacked_state).unsqueeze(0).to(device))
                P = target.policy_network(output_p).cpu()
                P = P.squeeze(0).detach().numpy()

            if params['draw_image'] and global_i % params['draw_per_episode'] == 0:
                img = draw_pi(img, P)
                with torch.no_grad():
                    V = target.value_network(output_v)
                img = draw_val(img, to_scalar(V).squeeze(0).item())
                writer.add_image(f"image/episode_from_selfplay[{global_i}]", img, i, dataformats='HWC')

            if params['draw_wrapped_state']:
                writer.add_image(f"image/wrapped_state[{global_i}]", state_image, i, dataformats='CHW')
            
            action = np.random.choice(np.arange(params['action_space'] ), p=P)
            state, r, terminated, truncated, _ = self.env.step(action)   

            state_image = cv2.resize(state['pixels'], (params['resize_width'], params['resize_height']), interpolation=cv2.INTER_AREA).transpose(2,0,1)

            if params['stack_action_plane']:
                previous_action_plane = np.full((1, params['input_height'], params['input_width']), action/params['action_space']*params['norm_factor'])
                state = np.vstack((state_image, previous_action_plane))
            else:
                state = state_image

            self.action_traj.append(action)
            self.P_traj.append(P)
            
            if params['negative_reward'] and i==max_timestep-1 and not terminated:
                r = params['negative_reward_val']
            self.r_traj.append(r)
            
            game_score += r

            if terminated or i==max_timestep-1: # or truncated:
                last_frame = i
                if params['draw_image'] and global_i % params['draw_per_episode'] == 0:
                    img = draw_epi_act_rew(self.env.render(), episode_num=i+1, action=action, reward=r, score=game_score)
                    writer.add_image(f"image/episode_from_selfplay[{global_i}]", img, i+1, dataformats='HWC')
                break

        #print('self_play: score, r, done, info, lastframe', int(game_score), r, done, info, i)


        # for update inference over trajectory length
        for _ in range(params['unroll_step']+1):
            self.state_traj.append(np.zeros_like(state))
            self.P_traj.append(np.zeros_like(P))
            self.r_traj.append(0.0)
            self.action_traj.append(-1)  


        # traj append to replay
        self.state_replay.append(self.state_traj)
        self.action_replay.append(self.action_traj)
        self.P_replay.append(self.P_traj)
        self.r_replay.append(self.r_traj)  
        
        writer.add_scalar('Selfplay/score', game_score, global_i)
        writer.add_scalar('Selfplay/last_reward', r, global_i)
        writer.add_scalar('Selfplay/last_frame', last_frame, global_i)
        
        return game_score , r, last_frame


    def update_weights_mu(self, target):

        uniform_dist = torch.full((params['mb_dim'], params['action_space']), 1/params['action_space'], device=device)
        batch_action = torch.cat([torch.zeros(params['mb_dim'], 1) + i for i in range(params['action_space'])]).unsqueeze(0).to(device)
        
        for _ in range(params['iteration']): 
            state_traj = []
            action_traj = []
            P_traj = []
            r_traj = []      
            G_arr_mb = []

            for epi_sel in range(params['mb_dim']):
                if(epi_sel < params['mb_dim'] * params['replay_proportion'] / 100):
                    sel = np.random.randint(0,len(self.state_replay)) 
                else:
                    sel = -1

                ## multi step return G (orignally retrace used)
                G = 0
                G_arr = []
                for r in self.r_replay[sel][::-1]:
                    G = params['discount'] * G + r
                    G_arr.append(G)
                G_arr.reverse()
                
                for i in np.random.randint(len(self.state_replay[sel])-params['unroll_step']-1-params['stacking_frame']+1,size=1):
                    state_traj.append(self.state_replay[sel][i:i+params['unroll_step']+1+params['stacking_frame']-1]) 
                    action_traj.append(self.action_replay[sel][i:i+params['unroll_step']])
                    r_traj.append(self.r_replay[sel][i:i+params['unroll_step']])
                    G_arr_mb.append(G_arr[i:i+params['unroll_step']+1])                        
                    P_traj.append(self.P_replay[sel][i:i+params['unroll_step']+1])


            state_traj = torch.from_numpy(np.array(state_traj)).to(device)
            action_traj = torch.from_numpy(np.array(action_traj)).unsqueeze(2).to(device)
            P_traj = torch.from_numpy(np.array(P_traj)).to(device)
            G_arr_mb = torch.from_numpy(np.array(G_arr_mb)).unsqueeze(2).float().to(device)
            r_traj = torch.from_numpy(np.array(r_traj)).unsqueeze(2).float().to(device)
            inferenced_P_arr = [] 
            inferenced_r_logit_arr = []
            inferenced_v_logit_arr = []

            ## stacking frame
            stacked_state_0 = torch.cat([state_traj[:, i] for i in range(params['stacking_frame'])], dim=1)
            
            ## agent network inference (4 step unroll)            
            output_p, output_v, hs = self.representation_network(stacked_state_0)
            first_P, first_v_logits = self.policy_network(output_p), self.value_network(output_v)
            inferenced_P_arr.append(first_P)
            inferenced_v_logit_arr.append(first_v_logits)

            pre_p, pre_v, pre_r = self.dynamics_network(action_traj.transpose(0,1), hs)
            P, v_logits, r_logits = self.policy_network(pre_p), self.value_network(pre_v), self.reward_network(pre_r)
            for i in range(params['unroll_step']):
                inferenced_P_arr.append(P[i])
                inferenced_v_logit_arr.append(v_logits[i])
                inferenced_r_logit_arr.append(r_logits[i])
            
            ## target network inference
            with torch.no_grad():
                output_p, output_v, _ = target.representation_network(stacked_state_0)
                t_first_P, t_first_v_logits = target.policy_network(output_p), target.value_network(output_v)

            ## normalized advantage
            with torch.no_grad():
                beta_var = params['beta_var']
                self.var = beta_var*self.var + (1-beta_var)*(torch.sum((G_arr_mb[:,0] - to_scalar(t_first_v_logits))**2)/params['mb_dim'])
                self.beta_product *= beta_var
                var_hat = self.var/(1-self.beta_product)
                under = torch.sqrt(var_hat + params['eps_var'])

            ## L_pg_cmpo first term (eq.10)
            importance_weight = torch.clip(first_P.gather(1,action_traj[:,0])
                                        /(P_traj[:,0].gather(1,action_traj[:,0])),
                                        0, 1
            )
            first_term = -1 * importance_weight * (G_arr_mb[:,0] - to_scalar(t_first_v_logits))/under        


            ## second_term(exact KL) + L_m (now just L_m like Ada)
            L_m = 0     
            kl_loss = torch.nn.KLDivLoss(reduction="none")

            
            for i in range(params['unroll_step']+1):
                with torch.no_grad():
                    stacked_state = torch.cat([state_traj[:, i] for i in range(params['stacking_frame'])], dim=1)
                
                    output_p, output_v, t_hs = target.representation_network(stacked_state)
                    t_P, t_v_logits = target.policy_network(output_p), target.value_network(output_v)
                                        
                    batch_t_hs = (t_hs[0].repeat(1,params['action_space'],1), t_hs[1].repeat(1,params['action_space'],1))

                    _, pre_v, pre_r = target.dynamics_network(batch_action, batch_t_hs)
                    pre_v, pre_r = pre_v.squeeze(0), pre_r.squeeze(0)
                    batch_lookahead_v1, batch_lookahead_r1 = target.value_network(pre_v), target.reward_network(pre_r)

                    ## normalized advantage
                    beta_var = params['beta_var']
                    self.var_m[i] = beta_var*self.var_m[i] + (1-beta_var)*(torch.sum((G_arr_mb[:,i] - to_scalar(t_v_logits))**2)/params['mb_dim'])
                    self.beta_product_m[i]  *= beta_var
                    var_hat = self.var_m[i] /(1-self.beta_product_m[i])
                    under = torch.sqrt(var_hat + params['eps_var'])

                    adv = (to_scalar(batch_lookahead_r1) + params['discount']*to_scalar(batch_lookahead_v1) - to_scalar(t_v_logits).repeat(params['action_space'], 1))/under
                    exp_clip_adv = torch.exp(torch.clip(adv,-params['adv_clip_val'],params['adv_clip_val']))
                    
                    ## Paper appendix F.2 : Prior policy
                    if params['mixed_prior']:
                        t_P = 0.997*t_P + 0.003*uniform_dist # + 0.003*P_traj[:,i]
                    
                    pi_cmpo_all = t_P *exp_clip_adv.view(params['action_space'],params['mb_dim']).transpose(0,1)
                    pi_cmpo_all = pi_cmpo_all / torch.sum(pi_cmpo_all, dim=-1, keepdim=True)
                    
                L_m += kl_loss(torch.log(inferenced_P_arr[i]), pi_cmpo_all).sum(-1, keepdim=True)
                if(i==0):
                    L_m *= params['second_term_weight']
                                        
            if params['unroll_step'] > 0:
                L_m/=params['unroll_step']+1

            
            ## L_v, L_r            
            L_v = 0
            L_r = 0            
            for i in range(params['unroll_step']+1):
                L_v += (to_cr(G_arr_mb[:,i])*torch.log(inferenced_v_logit_arr[i]+1e-12)).sum(-1, keepdim=True)
            for i in range(params['unroll_step']):
                L_r += (to_cr(r_traj[:,i])*torch.log(inferenced_r_logit_arr[i]+1e-12)).sum(-1, keepdim=True)
            L_v*=-1
            L_v /= params['unroll_step']+1
            L_r*=-1
            if params['unroll_step'] > 0:
                L_r /= params['unroll_step']


            ## total loss
            L_total = params['total_policy_loss_weight']*(first_term + L_m) + params['value_loss_weight']*L_v + params['reward_loss_weight']*L_r 
            
            ## optimize
            self.optimizer.zero_grad()
            L_total.mean().backward()            
            nn.utils.clip_grad_value_(self.parameters(), clip_value=1.0)
            self.optimizer.step()
            
            ## target network(prior parameters) moving average update
            alpha_target = params['alpha_target']
            params1 = self.named_parameters()
            params2 = target.named_parameters()
            dict_params2 = dict(params2)
            for name1, param1 in params1:
                if name1 in dict_params2:
                    dict_params2[name1].data.copy_(alpha_target*param1.data + (1-alpha_target)*dict_params2[name1].data)
            target.load_state_dict(dict_params2)
            

        self.scheduler.step()
        
        writer.add_scalars('Loss(raw)',{'L_total': L_total.mean(),
                                  'first_term': (first_term).mean(),
                                  'L_m': (L_m).mean(),
                                  'L_v': (L_v).mean(),
                                  'L_r': (L_r).mean()
                                  },global_i)

        writer.add_scalars('under',{'under.mean': under.mean(),
                                         'under.max': under.max(),
                                         'under.min': under.min(),
                                  },global_i)
        
        writer.add_scalars('normed_adv',{'adv.mean': adv.mean(),
                                  'adv.max': adv.max(),
                                  'adv.min': adv.min(),
                                  },global_i)
        
        writer.add_scalars('output_embedding',{'mean(p)': output_p.mean(),
                                  'max(p)': output_p.max(),
                                  'min(p)': output_p.min(),
                                  'mean(v)': output_v.mean(),
                                  'max(v)': output_v.max(),
                                  'min(v)': output_v.min(),
                                  },global_i)
        
        return

#torch.manual_seed(42)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.benchmark = True
#np.random.seed(42)
#lstm related; CUBLAS_WORKSPACE_CONFIG=:16:8

print(torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
score_arr = []

agent = Agent()
target = Target()
print(agent)

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
if args.debug:
    writer = SummaryWriter()
else:
    log_dir = os.path.join(os.environ["PWD"], 'nni-experiments', os.environ["NNI_EXP_ID"], 'trials', os.environ["NNI_TRIAL_JOB_ID"], 'output/tensorboard')
    writer = SummaryWriter(log_dir)
    print(log_dir)


## initialization
target.load_state_dict(agent.state_dict())

## Self play & Weight update loop

for i in range(params['expriment_length']): 
    global_i = i            
    with debug_time("selfplay_time", global_i):
        game_score , last_r, frame = agent.self_play_mu(target)       
    nni.report_intermediate_result(game_score)
    score_arr.append(game_score)      
    if game_score > params['success_threshold'] and np.mean(np.array(score_arr[-20:])) > params['success_threshold']:
        print('Successfully learned')
        nni.report_final_result(game_score)
        break
    with debug_time("weight_update_time", global_i):
        agent.update_weights_mu(target) 

torch.save(target.state_dict(), 'weights_target.pt')  
agent.env.close()
writer.close()
