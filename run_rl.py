#import all these module 
import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import time
from uofgsocsai import *
from more_itertools import chunked

# Setup the parameters for the specific problem (you can change all of these if you want to) 
problem_id = 7   # problem_id \in [0:7] generates 8 diffrent problems on which you can train/fine-tune your agent 
reward_hole = -0.01     # should be less than or equal to 0.0 (you can fine tune this  depending on you RL agent choice)
is_stochastic = True  # should be False for A-star (deterministic search) and True for the RL agent

# Generate the specific problem 
env = LochLomondEnv(problem_id=problem_id, is_stochastic=True,   reward_hole=reward_hole)

# Let's visualize the problem/env
print(env.desc)

#create Q-table
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros((state_space_size, action_space_size))
#print(q_table)

# Reset the random generator to a known state (for reproducability)
np.random.seed(12)

#initialize parameters
max_episodes = 20000
max_iter_per_episode =2000
learning_rate = 0.1
gamma = 0.95

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

#start q learning

rewards = []
episode_step = []

for episode in range(max_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
#choose action   
    for step in range(max_iter_per_episode):
        exp_exp_tradeoff = random.uniform(0, 1)
        
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()
            
        new_state, reward, done, info = env.step(action)
        
#update Q table        
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma*np.max(q_table[new_state, :]) - q_table[state,action])
    
        total_rewards += reward        
        state = new_state
      
        if done == True:
            break
            
#update epsilon        
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    rewards.append(total_rewards)
    episode_step.append(step)
    
print("Score over time: " +  str(sum(rewards)/max_episodes))

#print(q_table)

#calculating & printing the average reward per thousand episodes
average_rewards =[sum(x) / len(x) for x in chunked(rewards, 100)]
#print("********Average rewards per 100 episodes********\n")
#print(average_rewards)

#calculating & printing the average reward per thousand episodes
average_steps =[sum(x) / len(x) for x in chunked(episode_step, 100)]


#save results
filename = "out_rl_qtable"
title ="\n---------------------------\n" "\n Q table of problem "+str(problem_id)+"\n"
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
content =q_table
with open(filename+".txt", "a") as f:
   f.write(str(title)+str(content))

#plot1
   
imgname = "RL_plot_"+str(problem_id)

#set name of x,y
plt.title(imgname)
plt.grid()
plt.xlabel("episode x100")
plt.ylabel('Average reward over 100 episodes')
plt.plot(average_rewards)
plt.savefig(imgname)
plt.show()

#plot2

imgname = "RL_plot_step"+str(problem_id)

#set name of x,y
plt.title(imgname)
plt.grid()
plt.xlabel("episode x100")
plt.ylabel('Average steps over 100 episodes')
plt.plot(average_steps)
plt.savefig(imgname)
plt.show()


