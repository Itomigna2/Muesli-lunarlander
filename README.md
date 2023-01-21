# Muesli (LunarLander-v2)

## Introduction

Here is simple implementation of Muesli algorithm. Muesli has same performance and network architecture as MuZero, but it can be trained without MCTS lookahead search, just use one-step lookahead. It can reduce computational cost significantly compared to MuZero.

Paper : [Muesli: Combining Improvements in Policy Optimization, Hessel et al., 2021](https://arxiv.org/abs/2104.06159) (v2 version)

You can run this code on [colab demo link](https://colab.research.google.com/drive/1h3Xy1AFn_CEgvKZkS8E2xasHDm5p03Xb?usp=sharing), train the agent and monitor with tensorboard, play LunarLander-v2 environment with trained network. This agent can solve LunarLander-v2 within 1~2 hours computed by Google Colab CPU backend. It can reach about > 250 average score.


## Implemented
- [x] MuZero network
- [x] 5 step unroll
- [x] L_pg+cmpo
- [x] L_v
- [x] L_r
- [x] L_m (5 step)
- [x] Stacking 8 observations
- [x] Mini-batch update 
- [x] Hidden state scaled within [-1,1]
- [x] Gradient clipping by value [-1,1]
- [x] Dynamics network gradient scale 1/2
- [x] Target network(prior parameters) moving average update
- [x] Categorical representation (value, reward model)
- [x] Normalized advantage
- [x] Tensorboard monitoring

## Todo
- [ ] Retrace estimator 
- [ ] CNN representation network
- [ ] LSTM dynamics network
- [ ] Atari environment

## Differences from paper
- [x] Self-play use agent network (originally target network)

## Self-play
Flow of self-play.
![selfplay3](https://user-images.githubusercontent.com/119741210/213879476-651f13f8-dc70-4033-b9f6-13efbe81bcc5.png)

## Unroll structure
Target network 1-step unroll : When calculating v_pi_prior(s) and second term of L_pg+cmpo.

Unroll 5-step(agent network) : Unroll agent network to optimize.

1-step unrolls for L_m (target network) : When calculating pi_cmpo of L_m.
![Unroll](https://user-images.githubusercontent.com/119741210/213876179-62566fbc-dbce-4edb-9e56-d031e43e1e29.png)

## Results
Score graph
![score](https://user-images.githubusercontent.com/119741210/213872123-b306563a-0a04-4fcc-815c-3f04cac01e0a.png)
Loss graph
![loss](https://user-images.githubusercontent.com/119741210/213872175-5ce19b30-836b-45a8-bfc1-371598a27b03.png)
Lunarlander play length and last rewards
![lastframe_lastreward](https://user-images.githubusercontent.com/119741210/213876120-167c9211-a3ae-42a6-90c3-0f93279cec7c.png)
Var variables of advantage normalization
![var](https://user-images.githubusercontent.com/119741210/213876126-936c0098-e021-42da-b97c-615360f20bba.png)

## Comment
Need your help! Welcome to contribute, advice, question, etc.

Contact : emtgit2@gmail.com (Available languages : English, Korean)

## Links
Lunarlander-v2 env document : https://www.gymlibrary.dev/environments/box2d/lunar_lander/
