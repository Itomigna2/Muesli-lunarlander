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

import torchsummary


params = {
    ## Params controlled by this file
    'game_name': 'LunarLander-v2', # gym env name
    'actor_max_epi_len': 1000, # max step of gym env
    'success_threshold': 200, # arbitrary success threshold of game score
    
    'RGB_Wrapper': True, # change vector based state to RGB
    #'stack_action_plane': True, # stack action information plane to RGB state
    'norm_factor': 255.0, # normalize RGB by /255.0
    'resizing_state': True, # overwrite the H,W related params to resize

    
    'draw_wrapped_state': False, # draw image which agent see actually
    'draw_image': True, # draw full resolution RGB episode
    'draw_per_episode': 50, # drawing slows donw the code, adjust frequency by this
    'negative_reward': False, # experimental feature, making last zero reward to negative reward
    'negative_reward_val': -100.0, # negative reward value
    

    
    'unroll_step' : 4, # unroll step
    'beta_var': 0.99, # related to advantage normalization
    'eps_var': 1e-12, # related to advantage normalization
    'hs_resolution': 36, # resolution of hidden state
    'mlp_width': 128, # mlp network width
    'support_size': 30, # support_size of categorical representation
    'eps': 0.001, # categorical representation related
    'discount': 0.995, # discount rate
    'stacking_frame': 8, # stacking previous states
    'value_loss_weight': 0.25, # multiplier for value loss
    'reward_loss_weight': 1, # multiplier for reward loss

    
    ## HPO params controlled by config.yaml
    'regularizer_multiplier': 1, # multiplier of regularization term
    'alpha_target': 0.01, # target network(prior parameters) moving average update ratio
    
    'mb_dim': 128, # dimension of minibatch 
    'iteration': 80, # num of iteration 
    'replay_proportion': 75, # proportion of the replay inside minibatch 
    'start_lr': 0.0003, # learning rate
    'expriment_length': 4000, # num of repetitions of self-play&update
    'policy_loss_weight': 1, # multiplier for policy loss

    'resize_height': 72, # image resize H
    'resize_width': 96, # image resize W

    'stack_action_plane': False, # stack action information plane to RGB state
    
    ## Params will be assigned by the code
        # params['input_height']
        # params['input_width']
        # params['input_channels']
}


optimized_params = nni.get_next_parameter()
params.update(optimized_params)

## For linear lr decay
## https://github.com/cmpark0126/pytorch-polynomial-lr-decay
from torch_poly_lr_decay import PolynomialLRDecay


class Representation(nn.Module): 
    def __init__(self, input_channels, hidden_size, width):
        super().__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, (8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, (2, 2), stride=2),
            nn.ReLU()
        )
        self.fc = nn.Linear(240, hidden_size)

    def forward(self, x):
        x = x.div(params['norm_factor']).float()
        x = self.image_conv(x)
        x = torch.flatten(x, start_dim=1)  
        x = F.relu(self.fc(x))
        x = (x - x.min(-1,keepdim=True)[0])/(x.max(-1,keepdim=True)[0] - x.min(-1,keepdim=True)[0])
        return x


class Dynamics(nn.Module): 
    def __init__(self, input_dim, output_dim, width):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim + params['action_space'], width)
        self.layer2 = torch.nn.Linear(width, width) 
        self.hs_head = torch.nn.Linear(width, output_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(width,width),
            nn.ReLU(),
            nn.Linear(width,width),
            nn.ReLU(),
            nn.Linear(width,support_size*2+1)           
        ) 
        self.one_hot_act = torch.cat((torch.nn.functional.one_hot(torch.arange(0, params['action_space']) % params['action_space'], num_classes=params['action_space']), torch.zeros(params['action_space']).unsqueeze(0)), dim=0).to(device)        
        
    def forward(self, x, action):
        if(action.dim()==2):
            action = self.one_hot_act[action.squeeze(1)]
            x = torch.cat((x,action.to(device)), dim=1)
            x = self.layer1(x)
            x = torch.nn.functional.relu(x)
            x = self.layer2(x)
            x = torch.nn.functional.relu(x)
            hs = self.hs_head(x)
            hs = torch.nn.functional.relu(hs)
            reward = self.reward_head(x)    
            hs = (hs - hs.min(-1,keepdim=True)[0])/(hs.max(-1,keepdim=True)[0] - hs.min(-1,keepdim=True)[0])
        if(action.dim()==3):
            action = torch.nn.functional.one_hot(action.to(torch.int64), num_classes=params['action_space'])
            action = action.squeeze(2)
            x = torch.cat((x,action.to(device)), dim=2)
            x = self.layer1(x)
            x = torch.nn.functional.relu(x)
            x = self.layer2(x)
            x = torch.nn.functional.relu(x)
            hs = self.hs_head(x)
            hs = torch.nn.functional.relu(hs)
            reward = self.reward_head(x)    
            hs = (hs - hs.min(-1,keepdim=True)[0])/(hs.max(-1,keepdim=True)[0] - hs.min(-1,keepdim=True)[0])
        return hs, reward


class Prediction(nn.Module): 
    def __init__(self, input_dim, width):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, width)
        self.layer2 = torch.nn.Linear(width, width) 
        self.policy_head = nn.Sequential(
            nn.Linear(width,width),
            nn.ReLU(),
            nn.Linear(width,width),
            nn.ReLU(),
            nn.Linear(width, params['action_space'])           
        ) 
        self.value_head = nn.Sequential(
            nn.Linear(width,width),
            nn.ReLU(),
            nn.Linear(width,width),
            nn.ReLU(),
            nn.Linear(width,support_size*2+1)           
        ) 
   
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)
        P = self.policy_head(x)
        P = torch.nn.functional.softmax(P, dim=-1) 
        V = self.value_head(x)      
        return P, V


support_size = params['support_size']
eps = params['eps']

def to_scalar(x):
    x = torch.softmax(x, dim=-1)
    probabilities = x
    support = (torch.tensor([x for x in range(-support_size, support_size + 1)]).expand(probabilities.shape).float().to(device))
    x = torch.sum(support * probabilities, dim=1, keepdim=True)
    scalar = torch.sign(x) * (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps))** 2 - 1)
    return scalar

def to_scalar_3D(x):
    x = torch.softmax(x, dim=-1)
    probabilities = x
    support = (torch.tensor([x for x in range(-support_size, support_size + 1)]).expand(probabilities.shape).float().to(device))
    x = torch.sum(support * probabilities, dim=2, keepdim=True)
    scalar = torch.sign(x) * (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps))** 2 - 1)
    return scalar

def to_scalar_no_soft(x): ## test purpose
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




##Target network
class Target(nn.Module):
    """Target Network
    
    Target network is used to approximate v_pi_prior, q_pi_prior, pi_prior.
    It contains older network parameters. (exponential moving average update)
    """
    def __init__(self, width):
        super().__init__()
        self.representation_network = Representation(params['stacking_frame']*params['input_channels'], params['hs_resolution'], width) 
        self.dynamics_network = Dynamics(params['hs_resolution'], params['hs_resolution'], width)
        self.prediction_network = Prediction(params['hs_resolution'], width)  
        self.to(device)


##Muesli agent
class Agent(nn.Module):
    """Agent Class"""
    def __init__(self, width):
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
        
        self.representation_network = Representation(params['stacking_frame']*params['input_channels'], params['hs_resolution'], width) 
        self.dynamics_network = Dynamics(params['hs_resolution'], params['hs_resolution'], width)
        self.prediction_network = Prediction(params['hs_resolution'], width) 
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
        action = -1
        r = -1
        last_frame = 1000

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
                hs = target.representation_network(torch.from_numpy(stacked_state).float().unsqueeze(0).to(device))
                P, v = target.prediction_network(hs)    
                P = P.squeeze(0)

            if params['draw_image'] and global_i % params['draw_per_episode'] == 0:
                img = draw_pi(img, P.detach().cpu().numpy())
                writer.add_image(f"image/episode_from_selfplay[{global_i}]", img, i, dataformats='HWC')

            if params['draw_wrapped_state']:
                writer.add_image(f"image/wrapped_state[{global_i}]", state_image, i, dataformats='CHW')
            
            action = np.random.choice(np.arange(params['action_space'] ), p=P.detach().cpu().numpy())   
            state, r, terminated, truncated, _ = self.env.step(action)   

            state_image = cv2.resize(state['pixels'], (params['resize_width'], params['resize_height']), interpolation=cv2.INTER_AREA).transpose(2,0,1)

            if params['stack_action_plane']:
                previous_action_plane = np.full((1, params['input_height'], params['input_width']), action/params['action_space']*params['norm_factor'])
                state = np.vstack((state_image, previous_action_plane))
            else:
                state = state_image


            self.action_traj.append(action)
            self.P_traj.append(P.cpu().numpy())
            
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
                    state_traj.append(self.state_replay[sel][i:i+params['unroll_step']+1+params['stacking_frame']]) 
                    action_traj.append(self.action_replay[sel][i:i+params['unroll_step']+1])
                    r_traj.append(self.r_replay[sel][i:i+params['unroll_step']+1])
                    G_arr_mb.append(G_arr[i:i+params['unroll_step']+1])                        
                    P_traj.append(self.P_replay[sel][i])


            state_traj = torch.from_numpy(np.array(state_traj)).to(device)
            action_traj = torch.from_numpy(np.array(action_traj)).unsqueeze(2).to(device)
            P_traj = torch.from_numpy(np.array(P_traj)).to(device)
            G_arr_mb = torch.from_numpy(np.array(G_arr_mb)).unsqueeze(2).float().to(device)
            r_traj = torch.from_numpy(np.array(r_traj)).unsqueeze(2).float().to(device)
            inferenced_P_arr = []
            inferenced_r_logit_arr = []
            inferenced_v_logit_arr = []

            ## stacking 8 frame
            stacked_state_0 = torch.cat([state_traj[:, i] for i in range(params['stacking_frame'])], dim=1)
            
            ## agent network inference (5 step unroll)            
            hs = self.representation_network(stacked_state_0)
            first_P, first_v_logits = self.prediction_network(hs)
            hs.register_hook(lambda grad: grad * 0.5)
            inferenced_P_arr.append(first_P)
            inferenced_v_logit_arr.append(first_v_logits)            

            for i in range(params['unroll_step']):
                hs, r_logits = self.dynamics_network(hs, action_traj[:,i])    
                P, v_logits = self.prediction_network(hs)
                hs.register_hook(lambda grad: grad * 0.5)
                inferenced_P_arr.append(P)            
                inferenced_r_logit_arr.append(r_logits)
                inferenced_v_logit_arr.append(v_logits)

            ## target network inference
            with torch.no_grad():
                t_first_hs = target.representation_network(stacked_state_0)
                t_first_P, t_first_v_logits = target.prediction_network(t_first_hs)  

            ## normalized advantage
            beta_var = params['beta_var']
            self.var = beta_var*self.var + (1-beta_var)*(torch.sum((G_arr_mb[:,0] - to_scalar(t_first_v_logits))**2)/params['mb_dim'])
            self.beta_product *= beta_var
            var_hat = self.var/(1-self.beta_product)
            under = torch.sqrt(var_hat + params['eps_var'])

            ## L_pg_cmpo first term (eq.10)
            importance_weight = torch.clip(first_P.gather(1,action_traj[:,0])
                                        /(P_traj.gather(1,action_traj[:,0])),
                                        0, 1
            )
            first_term = -1 * importance_weight * (G_arr_mb[:,0] - to_scalar(t_first_v_logits))/under        

            ## second_term(exact KL) + L_m

            L_m = 0      
            for i in range(params['unroll_step']+1):
                with torch.no_grad():
                    stacked_state = torch.cat([state_traj[:, i] for i in range(params['stacking_frame'])], dim=1)
                
                    t_hs = target.representation_network(stacked_state)
                    t_P, t_v_logits = target.prediction_network(t_hs)
                    
                    batch_t_hs = t_hs.repeat(params['action_space'] , 1, 1)                    
                    batch_action = torch.zeros(params['mb_dim'], 1, dtype=torch.int64)
                    batch_action = torch.cat([batch_action + i for i in range(params['action_space'])], dim=0).view(params['action_space'], params['mb_dim'], 1)

                    batch_lookahead_hs, batch_lookahead_r1 = target.dynamics_network(batch_t_hs, batch_action)
                    _, batch_lookahead_v1 = target.prediction_network(batch_lookahead_hs)

                    ## normalized advantage
                    beta_var = params['beta_var']
                    self.var_m[i] = beta_var*self.var_m[i] + (1-beta_var)*(torch.sum((G_arr_mb[:,i] - to_scalar(t_v_logits))**2)/params['mb_dim'])
                    self.beta_product_m[i]  *= beta_var
                    var_hat = self.var_m[i] /(1-self.beta_product_m[i])
                    under = torch.sqrt(var_hat + params['eps_var'])

                    exp_clip_adv = torch.exp(torch.clip((to_scalar_3D(batch_lookahead_r1) + params['discount']*to_scalar_3D(batch_lookahead_v1) - to_scalar(t_v_logits).repeat(params['action_space'], 1, 1))/under, -1,1))
                    
                    ## Paper appendix F.2 : Prior policy
                    t_P = 0.967*t_P + 0.03*P_traj + 0.003*torch.full((params['mb_dim'], params['action_space'] ), 1/params['action_space'], device=device)
                    
                    pi_cmpo_all = t_P * exp_clip_adv.transpose(0,1).squeeze(-1)
                    pi_cmpo_all = pi_cmpo_all / torch.sum(pi_cmpo_all, dim=-1, keepdim=True)         
                                        
                    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
                if(i==0):
                    second_term = kl_loss(torch.log(inferenced_P_arr[i]), pi_cmpo_all)
                else:
                    L_m += kl_loss(torch.log(inferenced_P_arr[i]), pi_cmpo_all) 

            if params['unroll_step'] > 0:
                L_m/=params['unroll_step']

            
            #writer.add_scalar(f"adv_norm/var_m[{i}]", self.var_m[i], idx)
            
            
            ## L_pg_cmpo               
            L_pg_cmpo = first_term + params['regularizer_multiplier'] * second_term
            
            
            ## L_v, L_r
            
            L_v = 0
            L_r = 0

            CEloss = nn.CrossEntropyLoss(reduction='none')
            
            for i in range(params['unroll_step']+1):
                L_v += CEloss(inferenced_v_logit_arr[i], to_cr(G_arr_mb[:,i])).unsqueeze(-1)
            
            for i in range(params['unroll_step']):
                L_r += CEloss(inferenced_r_logit_arr[i], to_cr(r_traj[:,i])).unsqueeze(-1)

            L_v /= params['unroll_step']+1
            if params['unroll_step'] > 0:
                L_r /= params['unroll_step']


            ## total loss
            L_total = (L_pg_cmpo + L_m)*params['policy_loss_weight']+ L_v*params['value_loss_weight'] + L_r*params['reward_loss_weight']   

            
            ## optimize
            self.optimizer.zero_grad()
            L_total.mean().backward()
            nn.utils.clip_grad_value_(self.parameters(), clip_value=1.0)
            self.optimizer.step()
            

            '''
            print('-------self.named_modules-------\n')
            for name, module in self.named_modules():
                print(name)
                #print(module)

            print('-------target.named_modules-------\n')
            for name, module in target.named_modules():
                print(name)
                #print(module)

            print('-------self.named_buffers-------\n')
            for name, module in self.named_buffers():
                print(name)
                #print(module)
            
            print('-------target.named_modules-------\n')
            for name, module in target.named_buffers():
                print('target:\n', name)
                #print(module)
            '''
            
            ## target network(prior parameters) moving average update
            alpha_target = params['alpha_target']
            params1 = self.named_parameters()
            params2 = target.named_parameters()
            dict_params2 = dict(params2)
            for name1, param1 in params1:
                if name1 in dict_params2:
                    dict_params2[name1].data.copy_(alpha_target*param1.data + (1-alpha_target)*dict_params2[name1].data)

            '''
            buffers1 = self.named_buffers()
            buffers2 = target.named_buffers()
            dict_buffers2 = dict(buffers2)
            for name1, buffer1 in buffers1:
                if name1 in dict_buffers2:
                    dict_buffers2[name1].data.copy_(alpha_target*buffer1.data + (1-alpha_target)*dict_buffers2[name1].data)
                    
            target.load_state_dict(dict(dict_params2, **dict_buffers2))          
            '''
            target.load_state_dict(dict_params2)

        self.scheduler.step()

        
        writer.add_scalars('Loss',{'L_total': L_total.mean(),
                                  'L_pg_cmpo': (L_pg_cmpo*params['policy_loss_weight']).mean(),
                                  'L_m': (L_m*params['policy_loss_weight']).mean(),
                                  'L_v': (L_v*params['value_loss_weight']).mean(),
                                  'L_r': (L_r*params['reward_loss_weight']).mean()
                                  },global_i)


        for i in range(params['unroll_step']+1):
            writer.add_scalar(f"adv_norm/var_m[{i}]", self.var_m[i], global_i)

        
        return


print(torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
score_arr = []


agent = Agent(params['mlp_width'])
target = Target(params['mlp_width'])
print(agent)
#torchsummary.summary(agent.representation_network, input_size=(24, 80, 120))

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
