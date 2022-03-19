#!/usr/bin/env python
import gym
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense , Activation
from keras.models import Sequential , load_model
from keras.optimizers import Adam
import rospy
import rospkg
from gym import wrappers
import irb120_env
def DNN (learning_rate , num_state , num_actions , layer_one , layer_two):

    calssifier = Sequential()
    calssifier.add(Dense(output_dim=layer_one,init='uniform',activation='relu',input_dim=num_state))
    calssifier.add(Dense(output_dim=layer_two,init='uniform',activation='relu'))
    calssifier.add(Dense(output_dim=num_actions,init='uniform'))
    calssifier.compile(optimizer=Adam(lr = learning_rate),loss='mse')
    return calssifier

    
    

class replay_buffer ():
    def __init__(self , memory_size , num_actions , num_states , discrete = True ):
        
        self.memory_size = memory_size
        self.num_actions = num_actions
        self.num_states = num_states
        self.discrete = discrete
        
        self.d_type = np.int8 if self.discrete else np.float32
        self.state_memory = np.zeros((self.memory_size , self.num_states))
        self.new_state_memory = np.zeros((self.memory_size , self.num_states))
        self.action_memory = np.zeros((self.memory_size , self.num_actions))
        self.reward_memory = np.zeros((self.memory_size ))
        self.terminal_memory = np.zeros(self.memory_size , dtype = np.float32)
        self.mem_count = 0
        
        
    def store_transition(self , state , action , reward , new_state , done):
        index = self.mem_count % self.memory_size
        
        self.state_memory[index] = state
        self. new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        actions = np.zeros( self.action_memory.shape[1] )
        actions[action] = 1
        self.action_memory[index] = actions
    
    
        self.mem_count +=1
        
    def sampler(self , batch_size):
        max_mem = min(self.mem_count , self.memory_size)
        batch = np.random.choice(max_mem , batch_size)
        
        state = self.state_memory[batch] 
        new_state = self. new_state_memory[batch] 
        reward = self.reward_memory[batch]
        actions =  self.action_memory[batch] 
        terminal = self.terminal_memory[batch] 
        
        return state , actions , reward , new_state  , terminal 
        
        
class Agent(object):
    
    def __init__(self ,learning_rate  , discount_factor , num_actions 
                 , epsilon , batch_size , num_states , epsilon_dec , epsilon_min , memory_size 
                 , layer_one , layer_two , model_name):
        
        self.action_space = [i for i in range (num_actions)]
        self.learning_rate  = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon 
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size 
        self.model_name = model_name
        self.memory =  replay_buffer(memory_size , num_actions , num_states , discrete = False)
        
        self.q_eval = DNN(learning_rate , num_states , num_actions  , layer_one , layer_two)
        
    def remember(self , state , action , reward , new_state , done):
        self.memory.store_transition(state , action , reward , new_state , done)
        
    def choose_action(self , state):
        rospy.loginfo("DQN STATE")
        #state = np.asarray(state,dtype=np.float64)
        rospy.loginfo(state)

        state = state[np.newaxis , :]
        rand = np.random.random()
        if rand < self.epsilon :
            action = np.random.choice(self.action_space)
        
        else:
            action = self.q_eval.predict(state)
            action = np.argmax(action)
            
        return action
        
        
        
    def learn(self):
        if self.memory.mem_count < self.batch_size:
            return 
        state , action , reward , new_state , done = self.memory.sampler(self.batch_size)
        action_values = np.array(self.action_space , dtype = np.int8)
        action = action.astype(int)
        action_index = np.dot(action , action_values)
        
        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)
        
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size , dtype = np.int32)        
                
        target = reward + (self.discount_factor*np.max(q_next , axis= 1)*done)
        #rospy.loginfo(np.shape(action))
        #rospy.loginfo(action)
       # rospy.loginfo(action_values)
        #rospy.loginfo(action_index)

        q_target[batch_index , action_index]  =  target
        
        _ = self.q_eval.fit(state , q_target , verbose = 0)        
        if self.epsilon > self.epsilon_min:
           self.epsilon = self.epsilon * self.epsilon_dec 
        else:
            self.epsilon = self.epsilon_min
        
    def save_model(self):
        self.q_eval.save_weights(self.model_name)
        
        
   # def load_modele(self):
      #  self.q_eval.load(self , self.model_name)
        
        
#####################################################################
###########################  MAIN  ##################################        
#####################################################################
        
        
if __name__ == '__main__':
    rospy.init_node('abb_irb120_gym', anonymous=True, log_level=rospy.INFO)
    env = gym.make('irb120-v0')
    layer_one = 256
    layer_two = 256
    learning_rate  = 0.01
    discount_factor = 0.99
    num_actions = 12
    epsilon = 1
    batch_size =64
    num_states = 3
    epsilon_dec = 0.999
    epsilon_min =0.01
    memory_size  = 1000000
    model_name = 'IRB_DQN_MODEL.h5'
    episodes = 100
    agent = Agent(learning_rate  , discount_factor , num_actions 
                 , epsilon , batch_size , num_states , epsilon_dec , epsilon_min , memory_size 
                 , layer_one , layer_two , model_name)
    
    
    scores = [] 
    avg_scores = [] 
    ep_history = []

    for i in range (episodes):
        
        done = False
        score = 0
        observation = env.reset()
        while not done :
            action = agent.choose_action(observation)
            rospy.loginfo("action")
            rospy.loginfo(action)
            new_observation , reward , done , info = env.step(action) 
            #print(new_observation)
            score += reward 
            agent.remember(observation , action , reward , new_observation , done)
            observation = new_observation
            agent.learn()
            ep_history.append(agent.epsilon)
        
        scores.append(score)
        avg_score = np.mean(scores[max(0,i-100):(i+1)])
        avg_scores.append(avg_score)
        print('episode', i , 'score' , score , 'average' , avg_score)
        plt.plot(avg_scores)

    agent.save_model()
    plt.show()
    env.close()
        
        
        
        
        
        
        
        
