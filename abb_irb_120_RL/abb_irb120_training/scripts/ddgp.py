#!/usr/bin/env python
import os
import gym
import rospy
import irb120_env_cont
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Action_Noise(object):
    
    def __init__(self , mu , sigma=0.15 , theta=0.2 , dt=1e-2 , x0 = None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()
        
    def __call__(self):
        x = self.x_prev + self.theta*(self.mu - self.x_prev)*self.dt + \
        self.sigma*np.sqrt(self.dt)*np.random.normal(size = self.mu.shape)
        
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev= self.x0 if self.x0 is not None else np.zeros_like(self.mu)           
        
        
        
            
class replay_buffer (object):
    
    def __init__(self , memory_size , num_actions , num_states ):
        
        self.memory_size = memory_size
        self.num_actions = num_actions
        self.num_states = num_states
        self.mem_count = 0
        
        
        self.state_memory = np.zeros((self.memory_size , num_states))
        self.new_state_memory = np.zeros((self.memory_size , num_states))
        self.action_memory = np.zeros((self.memory_size , self.num_actions))
        self.reward_memory = np.zeros((self.memory_size ))
        self.terminal_memory = np.zeros(self.memory_size , dtype = np.float32)
        
        
    def store_transition(self , state , action , reward , new_state , done):
        index = self.mem_count % self.memory_size
        
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self. new_state_memory[index] = new_state
        self.terminal_memory[index] = 1 - int(done)
        
        self.mem_count +=1
        
        
        
    def sampler(self , batch_size):
        max_mem = min(self.mem_count , self.memory_size)
        batch = np.random.choice(max_mem , batch_size)
        
        state = self.state_memory[batch]
        actions =  self.action_memory[batch]
        reward = self.reward_memory[batch]
        new_state = self. new_state_memory[batch] 
        terminal = self.terminal_memory[batch] 
        
        return state , actions , reward , new_state  , terminal
    
    
class Critic_Net(nn.Module):
    def __init__(self , lr , num_states , num_actions , l1_dim , l2_dim , name , chkpt_dir = 'tmp/ddpg'):
        super(Critic_Net , self).__init__()
    
        self.num_actions = num_actions
        self.num_states = num_states        
        self.layer_one = l1_dim
        self.layer_two = l2_dim
        self.learning_rate  = lr
        self.name = name
        self.checkpoint_file = os.path.join(chkpt_dir ,self.name +'_ddpg.ckpt')
        
        
        self.FC1 = nn.Linear(self.num_states , self.layer_one)
        f1= 1 / np.sqrt(self.FC1.weight.data.size()[0])
        T.nn.init.uniform_(self.FC1.weight.data , -f1 , f1)
        T.nn.init.uniform_(self.FC1.bias.data , -f1 , f1)
        self.bn1 = nn.LayerNorm(self.layer_one)

        self.FC2 = nn.Linear(self.layer_one , self.layer_two)
        f2= 1 / np.sqrt(self.FC2.weight.data.size()[0])
        T.nn.init.uniform_(self.FC1.weight.data , -f2 , f2)
        T.nn.init.uniform_(self.FC1.bias.data , -f2 , f2)
        self.bn2 = nn.LayerNorm(self.layer_two)
    
        self.action_value = nn.Linear(self.num_actions , self.layer_two)
        f3=0.003
        self.q =  nn.Linear(self.layer_two , 1)
        T.nn.init.uniform_(self.q.weight.data , -f3 , f3)
        T.nn.init.uniform_(self.q.bias.data , -f3 , f3)
        
        self.optimizer =optim.Adam(self.parameters() , lr = lr )
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, state , action):
        
        state_value = self.FC1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.FC2(state_value)
        state_value = self.bn2(state_value)
        
        action_value  = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value , action_value))
        state_action_value = self.q(state_action_value)
        
        return state_action_value
    
    
    def saver_checkpoint(self):
        print('------saving checkpoint-----')
        T.save(self.state_dict() , self.checkpoint_file)   
            
            
    def loader_checkpoint(self):
        print('------loading checkpoint-----')
        self.load_state_dict(T.load(self.checkpoint_file))
        
        
        
        
        
class Actor_Net(nn.Module):
    def __init__(self , lr , num_states , num_actions , l1_dim , l2_dim , name , chkpt_dir = 'tmp/ddpg'):
        super(Actor_Net , self).__init__()
    
        self.num_actions = num_actions
        self.num_states = num_states        
        self.layer_one = l1_dim
        self.layer_two = l2_dim
        self.learning_rate  = lr
        self.name = name
        self.checkpoint_file = os.path.join(chkpt_dir ,self.name +'_ddpg.ckpt')
        
        
        self.FC1 = nn.Linear(self.num_states , self.layer_one)
        f1= 1 / np.sqrt(self.FC1.weight.data.size()[0])
        T.nn.init.uniform_(self.FC1.weight.data , -f1 , f1)
        T.nn.init.uniform_(self.FC1.bias.data , -f1 , f1)
        self.bn1 = nn.LayerNorm(self.layer_one)

        self.FC2 = nn.Linear(self.layer_one , self.layer_two)
        f2= 1 / np.sqrt(self.FC2.weight.data.size()[0])
        T.nn.init.uniform_(self.FC1.weight.data , -f2 , f2)
        T.nn.init.uniform_(self.FC1.bias.data , -f2 , f2)
        self.bn2 = nn.LayerNorm(self.layer_two)
    
        f3=0.003
        self.mu =  nn.Linear(self.layer_two , self.num_actions)
        T.nn.init.uniform_(self.mu.weight.data , -f3 , f3)
        T.nn.init.uniform_(self.mu.bias.data , -f3 , f3)
        
        self.optimizer =optim.Adam(self.parameters() , lr = lr )
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, state ):
        
        x = self.FC1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.FC2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))
        
        return x
    
    
    def saver_checkpoint(self):
        print('------saving checkpoint-----')
        T.save(self.state_dict() , self.checkpoint_file)   
            
            
    def loader_checkpoint(self):
        print('------loading checkpoint-----')
        self.load_state_dict(T.load(self.checkpoint_file))
        
        
class Agent(object):
    
    def __init__(self ,lr_actor , lr_critic , discount_factor , num_actions ,
                 num_states , l1_dim , l2_dim , max_memory , tau , env , batch_size):
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.batch_size = batch_size
        self.memory=replay_buffer(max_memory , num_actions , num_states)
         
        self.actor = Actor_Net(lr_actor , num_states , num_actions , l1_dim , l2_dim , 'Actor')
        self.target_actor = Actor_Net(lr_actor , num_states , num_actions , l1_dim , l2_dim , 'T_Actor')
         
        self.critic = Critic_Net(lr_critic , num_states , num_actions , l1_dim , l2_dim , 'Critic')
        self.target_critic = Critic_Net(lr_critic , num_states , num_actions , l1_dim , l2_dim , 'T_Critic')
        
        
        self.noise = Action_Noise(mu = np.zeros(num_actions))
        
        self.update_network_params(tau = 1)
         
         
    def remember(self , state , action , reward , new_state , done):
        self.memory.store_transition(state , action , reward , new_state , done)

    
    def choose_action(self , state):
        
            self.actor.eval()
            state = T.tensor(state,dtype=T.float).to(self.actor.device)
            mu = self.actor(state).to(self.actor.device)
            mu_prime = mu + T.tensor(self.noise(),dtype=T.float).to(self.actor.device)
            
            self.actor.train()
            return mu_prime.cpu().detach().numpy()

    def learn(self):
        if self.memory.mem_count < self.batch_size:
            return 
        state , action , reward , new_state , done = self.memory.sampler(self.batch_size)

        state = T.tensor(state , dtype = T.float ).to(self.critic.device)        
        reward = T.tensor(reward , dtype = T.float ).to(self.critic.device)
        action = T.tensor(action , dtype = T.float ).to(self.critic.device)        
        new_state = T.tensor(new_state , dtype = T.float ).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(new_state)
        critic_val = self.target_critic.forward(new_state,target_actions)
        critic_value = self.critic.forward(state,action)


        target = []
        for i in range (self.batch_size):
            target.append(reward[i]+self.discount_factor*critic_val[i]*done[i])
            
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size ,1)
        
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target , critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        
        
        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state , mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        
        
        self.update_network_params()
        
    def update_network_params(self , tau = None):
        
        if tau is None:
            tau = self.tau
            
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)
        
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone()+\
             (1-tau)*target_critic_state_dict[name].clone()
            
            
            
        self.target_critic.load_state_dict(critic_state_dict)
        
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone()+\
             (1-tau)*target_actor_state_dict[name].clone()
            
        self.target_actor.load_state_dict(target_actor_state_dict)
        
        
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        


if __name__ == '__main__':
    rospy.init_node('abb_irb120_gym', anonymous=True, log_level=rospy.INFO)
    env = gym.make('irb120_cont-v0')
    learning_rate_actor  = 0.000025
    learning_rate_critic  = 0.00025
    discount_factor = 0.99
    num_actions = 6
    num_states = 3
    episodes = 100
    max_memory  = 1000000
    batch_size =64
    tau = 0.001
    layer_one = 400 
    layer_two = 300
    
    scores = [] 
    avg_scores = [] 
    ep_history = []
    
    np.random.seed(0)  
    
    agent = Agent(learning_rate_actor , learning_rate_critic , discount_factor , num_actions ,
                  num_states , layer_one , layer_two, max_memory , tau , env , batch_size)

               
    for i in range (episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done :
            action = agent.choose_action(observation)
            new_observation , reward , done , info = env.step(action) 
            agent.remember(observation , action , reward , new_observation , done)
            agent.learn()
            score += reward 
            observation = new_observation
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        print('episode', i , 'score' , score , 'average' , avg_score)
        plt.plot(avg_scores)
        
    env.close()
    plt.show()