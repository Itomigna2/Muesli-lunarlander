import math
import time

from tensorboardX import SummaryWriter
import gymnasium as gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

print(torch.cuda.is_available())

import nni
params = {
    'game_name': 'LunarLander-v2', 
    'env_observation_space': 8,
    'action_space': 4,
    
    'regularizer_multiplier': 1,
    'mb_dim': 128,
    'iteration': 80,
    'unroll_step' : 5,
    'stacking_frame': 8,
    'replay_proportion': 75,   # x%
    'start_lr': 0.0003,
    'expriment_length': 4000,
    'mlp_width': 128,
    'discount': 0.995,

    'support_size': 30,
    'eps': 0.001,

    'alpha_target': 0.01, # 0.1 ??
    'value_loss_weight': 0.25,
    'reward_loss_weight': 1,
    
    'beta_var': 0.99,
    'eps_var': 1e-12,

    'hs_resolution': 36,
        
    
}


optimized_params = nni.get_next_parameter()
params.update(optimized_params)


## For linear lr decay
## https://github.com/cmpark0126/pytorch-polynomial-lr-decay
from torch_poly_lr_decay import PolynomialLRDecay


class Representation(nn.Module): 
    """Representation Network

    Representation network produces hidden state from observations.
    Hidden state scaled within the bounds of [-1,1]. 
    Simple mlp network used with 1 skip connection.

    input : raw input
    output : hs(hidden state) 
    """
    def __init__(self, input_dim, output_dim, width):
        super().__init__()
        self.skip = torch.nn.Linear(input_dim, output_dim)  
        self.layer1 = torch.nn.Linear(input_dim, width)
        self.layer2 = torch.nn.Linear(width, width)
        self.layer3 = torch.nn.Linear(width, width) 
        self.layer4 = torch.nn.Linear(width, width)  
        self.layer5 = torch.nn.Linear(width, output_dim)     
        
    def forward(self, x):
        s = self.skip(x)
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)
        x = self.layer3(x)
        x = torch.nn.functional.relu(x)
        x = self.layer4(x)
        x = torch.nn.functional.relu(x)
        x = self.layer5(x)    
        x = torch.nn.functional.relu(x+s)
        x = 2*(x - x.min(-1,keepdim=True)[0])/(x.max(-1,keepdim=True)[0] - x.min(-1,keepdim=True)[0])-1 
        return x


class Dynamics(nn.Module): 
    """Dynamics Network

    Dynamics network transits (hidden state + action) to next hidden state and inferences reward model.
    Hidden state scaled within the bounds of [-1,1]. Action encoded to one-hot representation. 
    Zeros tensor is used for action -1.
    
    Output of the reward head is categorical representation, instaed of scalar value.
    Categorical output will be converted to scalar value with 'to_scalar()',and when 
    traning target value will be converted to categorical target with 'to_cr()'.
    
    input : hs, action
    output : next_hs, reward 
    """
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
            hs = 2*(hs - hs.min(-1,keepdim=True)[0])/(hs.max(-1,keepdim=True)[0] - hs.min(-1,keepdim=True)[0])-1
        if(action.dim()==3):
            action = torch.nn.functional.one_hot(action.to(torch.int64), num_classes=4)
            action = action.squeeze(2)
            x = torch.cat((x,action.to(device)), dim=2)
            x = self.layer1(x)
            x = torch.nn.functional.relu(x)
            x = self.layer2(x)
            x = torch.nn.functional.relu(x)
            hs = self.hs_head(x)
            hs = torch.nn.functional.relu(hs)
            reward = self.reward_head(x)    
            hs = 2*(hs - hs.min(-1,keepdim=True)[0])/(hs.max(-1,keepdim=True)[0] - hs.min(-1,keepdim=True)[0])-1
        return hs, reward


class Prediction(nn.Module): 
    """Prediction Network

    Prediction network inferences probability distribution of policy and value model from hidden state. 

    Output of the value head is categorical representation, instaed of scalar value.
    Categorical output will be converted to scalar value with 'to_scalar()',and when 
    traning target value will be converted to categorical target with 'to_cr()'.
        
    input : hs
    output : P, V 
    """
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


"""
For categorical representation
reference : https://github.com/werner-duvaud/muzero-general
In my opinion, support size have to cover the range of maximum absolute value of 
reward and value of entire trajectories. Support_size 30 can cover almost [-900,900].
"""
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


##Target network
class Target(nn.Module):
    """Target Network
    
    Target network is used to approximate v_pi_prior, q_pi_prior, pi_prior.
    It contains older network parameters. (exponential moving average update)
    """
    def __init__(self, state_dim, action_dim, width):
        super().__init__()
        self.representation_network = Representation(state_dim*params['stacking_frame'], params['hs_resolution'], width) 
        self.dynamics_network = Dynamics(params['hs_resolution'], params['hs_resolution'], width)
        self.prediction_network = Prediction(params['hs_resolution'], width)  
        self.to(device)


##Muesli agent
class Agent(nn.Module):
    """Agent Class"""
    def __init__(self, state_dim, action_dim, width):
        super().__init__()
        self.representation_network = Representation(state_dim*params['stacking_frame'], params['hs_resolution'], width) 
        self.dynamics_network = Dynamics(params['hs_resolution'], params['hs_resolution'], width)
        self.prediction_network = Prediction(params['hs_resolution'], width) 
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=params['start_lr'], weight_decay=0)
        self.scheduler = PolynomialLRDecay(self.optimizer, max_decay_steps=params['expriment_length'], end_learning_rate=0.0000)   
        self.to(device)

        self.state_replay = []
        self.action_replay = []
        self.P_replay = []
        self.r_replay = []   

        self.env = gym.make(params['game_name'])
                            
        self.var = 0
        self.beta_product = 1.0

        self.var_m = [0 for _ in range(params['unroll_step']+1)] 
        self.beta_product_m = [1.0 for _ in range(params['unroll_step']+1)] 


    def self_play_mu(self, target, max_timestep=1000):       
        """Self-play and save trajectory to replay buffer

        Self-play with target network parameter

        Eight previous observations stacked -> representation network -> prediction network 
        -> sampling action follow policy -> next env step
        """      

        self.state_traj = []
        self.action_traj = []
        self.P_traj = []
        self.r_traj = []      

        game_score = 0
        last_frame = 1000
        state = self.env.reset()
        state = state[0]
        state_dim = len(state)
        for i in range(max_timestep):   
            start_state = state
            if i == 0:
                stacked_state = np.tile(state, params['stacking_frame'])
            else:
                stacked_state = np.roll(stacked_state,-state_dim,axis=0)                
                stacked_state[-state_dim:]=state

            with torch.no_grad():
                hs = target.representation_network(torch.from_numpy(stacked_state).float().to(device))
                P, v = target.prediction_network(hs)    
            action = np.random.choice(np.arange(params['action_space'] ), p=P.detach().cpu().numpy())   
            state, r, done, info, _ = self.env.step(action)                    
            
            if i == 0:
                for _ in range(params['stacking_frame']):
                    self.state_traj.append(start_state)                
            else:
                self.state_traj.append(start_state)
            self.action_traj.append(action)
            self.P_traj.append(P.cpu().numpy())
            self.r_traj.append(r)
            
            game_score += r

            if done:
                last_frame = i
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
        '''
        writer.add_scalars('Selfplay',
                           {'lastreward': r,
                           # 'lastframe': last_frame+1
                           },global_i)
        '''
        return game_score , r, last_frame


    def update_weights_mu(self, target):
        """Optimize network weights.

        Iteration: 80
        Mini-batch size: 16 (1 seqeuence in 1 replay)
        Replay: 25% online data
        Discount: 0.997
        Unroll: 5 step
        L_m: 5 step(Muesli)
        Observations: Stack 8 frame
        regularizer_multiplier: 5 
        Loss: L_pg_cmpo + L_v/6/4 + L_r/5/1 + L_m
        """

        for _ in range(params['iteration']): 
            state_traj = []
            action_traj = []
            P_traj = []
            r_traj = []      
            G_arr_mb = []

            for epi_sel in range(params['mb_dim']):
                if(epi_sel < params['mb_dim'] * params['replay_proportion'] / 100):## replay proportion
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


            start = time.time()
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
                
            end = time.time()
            print(f"unroll time consume {end - start:.5f} sec")

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
            start = time.time()
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

            end = time.time()
            print(f"L_m time consume {end - start:.5f} sec")
            
            
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
            L_total = L_pg_cmpo + L_v*params['value_loss_weight'] + L_r*params['reward_loss_weight'] + L_m   

            
            start = time.time()
            ## optimize
            self.optimizer.zero_grad()
            L_total.mean().backward()
            nn.utils.clip_grad_value_(self.parameters(), clip_value=1.0)
            self.optimizer.step()
            end = time.time()
            print(f"optimize time consume {end - start:.5f} sec")
            

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

        '''
        writer.add_scalars('Loss',{'L_total': L_total.mean(),
                                  'L_pg_cmpo': L_pg_cmpo.mean(),
                                  'L_v': (L_v/6/4).mean(),
                                  'L_r': (L_r/5/1).mean(),
                                  'L_m': (L_m).mean()
                                  },global_i)
        
        writer.add_scalars('vars',{'self.var':self.var,
                                   'self.var_m':self.var_m[0]
                                  },global_i)
        '''
        
        return

#%rm -rf scalar/
#%load_ext tensorboard
#%tensorboard --logdir scalar --port=6010

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
score_arr = []

target = Target(params['env_observation_space'], params['action_space'] , params['mlp_width'])
agent = Agent(params['env_observation_space'], params['action_space'] , params['mlp_width'])  
print(agent)


## initialization
target.load_state_dict(agent.state_dict())

## Self play & Weight update loop

for i in range(params['expriment_length']):
    #writer = SummaryWriter(logdir='scalar/')
    global_i = i    
    start = time.time()
    game_score , last_r, frame = agent.self_play_mu(target)       
    end = time.time()
    print(f"selfplay time consume{end - start:.5f} sec")
    #writer.add_scalar('score', game_score, global_i)    
    nni.report_intermediate_result(game_score)
    score_arr.append(game_score)  
    print('episode, score, last_r, len\n', i, int(game_score), last_r, frame)
    
    if i%100==0:
        torch.save(target.state_dict(), 'weights_target.pt') 

    if game_score > 220 and np.mean(np.array(score_arr[-20:])) > 220:
        torch.save(target.state_dict(), 'weights_target.pt') 
        print('Done')
        nni.report_final_result(game_score)
        break
    start = time.time()
    agent.update_weights_mu(target) 
    end = time.time()
    print(f"update time consume {end - start:.5f} sec")
    #writer.close()

#nni.report_final_result(game_score)
torch.save(target.state_dict(), 'weights_target.pt')  
agent.env.close()



'''
## Earned score per episode

window = 30
mean_arr = []
for i in range(len(score_arr) - window + 1):
    mean_arr.append(np.mean(np.array(score_arr[i:i+window])))
for i in range(window - 1):
    mean_arr.insert(0, np.nan)

plt.plot(score_arr, label ='score')
plt.plot(mean_arr, label ='mean')
plt.ylim([-300,300])
plt.legend(loc='upper left')
plt.show()


## game play video(target network)
target.load_state_dict(torch.load("weights_target.pt"))
env = gnwrapper.LoopAnimation(gym.make(game_name)) 
state = env.reset()
state_dim = len(state)
game_score = 0
score_arr2 = []
state_arr = []
for i in range(1000):
    if i == 0:
        stacked_state = np.concatenate((state, state, state, state, state, state, state, state), axis=0)
    else:
        stacked_state = np.roll(stacked_state,-state_dim,axis=0)        
        stacked_state[-state_dim:]=state
    with torch.no_grad():
        hs = target.representation_network(torch.from_numpy(stacked_state).float().to(device))
        P, v = target.prediction_network(hs)
        action = np.random.choice(np.arange(agent.action_space), p=P.detach().numpy())   
    if i %5==0:
        env.render()
    state, r, done, info = env.step(action.item())
    print(r, done, info)
    state_arr.append(state[0])
    game_score += r 
    score_arr2.append(game_score)
    if i %10==0: 
        print(game_score)
    if done:
        print('last frame number : ',i)
        print('score :', game_score)
        print(state_arr)
       
        break
env.reset()
env.display()


plt.plot(score_arr2, label ='accumulated scores in one game play')
plt.legend(loc='upper left')
'''