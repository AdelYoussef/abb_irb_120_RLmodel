#!/usr/bin/env python
import gym
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Dense  , Input
from keras.models import Model
from keras.optimizers import Adam
import rospy
import irb120_env

class Agent(object):
    
    def __init__(self ,lr_actor , lr_critic , discount_factor , num_actions , num_states , layer_one , layer_two):
        
        self.action_space = [i for i in range (num_actions)]
        self.lr_actor  = lr_actor
        self.lr_critic  = lr_critic
        self.layer_one = layer_one
        self.layer_two = layer_two
        self.num_actions = num_actions
        self.num_states = num_states
        self.discount_factor = discount_factor
        
        self.actor , self.critic , self.policy =   self.build_actor_critic_network() 
        
    def build_actor_critic_network(self):
        
        INPUT = Input(shape = (self.num_states ,))
        delta = Input(shape =[1])
        dense1 = Dense(self.layer_one , activation = 'relu')(INPUT)
        dense2 = Dense(self.layer_two , activation = 'relu')(dense1)
        probs = Dense(self.num_actions , activation = 'softmax')(dense2) # actor outptu
        values = Dense(1, activation = 'linear')(dense2) # cricit output
        
        
         # y true is the action of the agent ,, y pred is the output of the network
        def custom_loss(y_true , y_pred):
            out = K.clip(y_pred , 1e-8 , 1-1e-8)
            log_lik = y_true*K.log(out) # liklelyhood 
            return K.sum(-log_lik*delta )
            
        
        actor = Model(input=[INPUT,delta],output=[probs])
        actor.compile(optimizer=Adam(lr=self.lr_actor),loss=custom_loss)   
        
        critic = Model(input=[INPUT],output=[values])
        critic.compile(optimizer=Adam(lr=self.lr_critic),loss='mse')    
            
        policy = Model(input=[INPUT],output=[probs])   
        return actor , critic , policy
            
            
    def choose_action(self , state):
        state = state[np.newaxis , :]
        probability = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space , p=probability)
        
        return action
            
            
            
    def learn(self , state , action , reward , new_state , done):
       
        state = state[np.newaxis , :]
        new_state = new_state[np.newaxis , :]
       
        critic_value_new = self.critic.predict(new_state)
        critic_value = self.critic.predict(state)
       
        target =  reward + self.discount_factor*critic_value_new*(1-int(done))
        delta = target - critic_value
       
        actions = np.zeros([1, self.num_actions])
        actions[np.arange(1),action] = 1
       
        self.actor.fit([state,delta],actions,verbose=0)
        self.critic.fit(state,target,verbose=0)
       
       
       
        
if __name__ == '__main__':
    rospy.init_node('abb_irb120_gym', anonymous=True, log_level=rospy.INFO)
    env = gym.make('irb120-v0')
    learning_rate_actor  = 0.00001
    learning_rate_critic  = 0.00005
    discount_factor = 0.99
    num_actions = 12
    num_states = 3
    episodes = 100
    scores = [] 
    avg_scores = [] 
    ep_history = []
    
    agent = Agent(learning_rate_actor , learning_rate_critic , discount_factor , num_actions ,
                  num_states , layer_one = 1024 , layer_two = 512)
       
       
    for i in range (episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done :
            action = agent.choose_action(observation)
            new_observation , reward , done , info = env.step(action) 
            #env.render()
            score += reward 
            agent.learn(observation , action , reward , new_observation , done )
            observation = new_observation
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        print('episode', i , 'score' , score , 'average' , avg_score)
        plt.plot(avg_scores)
        
    env.close()
    plt.show()
        
        
        
        
        
        
        
        
