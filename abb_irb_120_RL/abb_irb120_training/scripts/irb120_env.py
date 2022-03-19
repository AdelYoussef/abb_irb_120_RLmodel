import gym
import rospy
import time
import numpy as np
import math
import copy
import moveit_commander
from gym import utils, spaces
import numpy
from std_msgs.msg import Float64MultiArray , Float64
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock

from gazebo_connection import GazeboConnection
from controllers_connection import ControllersConnection

from gym.utils import seeding
from gym.envs.registration import register

reg = register(
    id='irb120-v0',
    entry_point='irb120_env:MYIRB120',
    max_episode_steps=100,
    )
class MYIRB120(gym.Env): # done
    def __init__(self):

        self.STEP =1

        robot= moveit_commander.RobotCommander()
        scene= moveit_commander.PlanningSceneInterface()

        self.arm=moveit_commander.MoveGroupCommander("manipulator")
        self.arm.set_pose_reference_frame("base_link")
        self.desired_position = [0.4,0.4,0.2]

        self.publishers_array = []
        self.current_ee = self.get_ee_pose()
        self.current_dist_from_des_pos_ee = self.calculate_distance_between(self.desired_position , self.current_ee)

        self._joint_1 = rospy.Publisher('/joint_1_position_controller/command', Float64, queue_size=1)
        self._joint_2 = rospy.Publisher('/joint_2_position_controller/command', Float64, queue_size=1)
        self._joint_3 = rospy.Publisher('/joint_3_position_controller/command', Float64, queue_size=1)
        self._joint_4 = rospy.Publisher('/joint_4_position_controller/command', Float64, queue_size=1)
        self._joint_5 = rospy.Publisher('/joint_5_position_controller/command', Float64, queue_size=1)
        self._joint_6 = rospy.Publisher('/joint_6_position_controller/command', Float64, queue_size=1)
            #self._gripper = rospy.Publisher("/gripper_controller/command", JointTrajectory, queue_size=1)
            #self._robotarm = rospy.Publisher("/arm_controller/command", JointTrajectory, queue_size=1)

        self.publishers_array.append(self._joint_1)
        self.publishers_array.append(self._joint_2)
        self.publishers_array.append(self._joint_3)
        self.publishers_array.append(self._joint_4)
        self.publishers_array.append(self._joint_5)
        self.publishers_array.append(self._joint_6)

        self.action_space = spaces.Discrete(12) #l,r,L,R,nothing
        self.seed()
        
        #get configuration parameters
        self.min_joint_1_angle = rospy.get_param('/irb120/min_joint_1_angle')
        self.min_joint_2_angle = rospy.get_param('/irb120/min_joint_2_angle')
        self.min_joint_3_angle = rospy.get_param('/irb120/min_joint_3_angle')
        self.min_joint_4_angle = rospy.get_param('/irb120/min_joint_4_angle')
        self.min_joint_5_angle = rospy.get_param('/irb120/min_joint_5_angle')
        self.min_joint_6_angle = rospy.get_param('/irb120/min_joint_6_angle')

        self.max_joint_1_angle = rospy.get_param('/irb120/max_joint_1_angle')
        self.max_joint_2_angle = rospy.get_param('/irb120/max_joint_2_angle')
        self.max_joint_3_angle = rospy.get_param('/irb120/max_joint_3_angle')
        self.max_joint_4_angle = rospy.get_param('/irb120/max_joint_4_angle')
        self.max_joint_5_angle = rospy.get_param('/irb120/max_joint_5_angle')
        self.max_joint_6_angle = rospy.get_param('/irb120/max_joint_6_angle')

        self.pos_step = rospy.get_param('/irb120/pos_step')
        self.running_step = rospy.get_param('/irb120/running_step')
        self.init_pos = rospy.get_param('/irb120/init_pos')
        self.wait_time = rospy.get_param('/irb120/wait_time')
        self.controllers_list = rospy.get_param('/irb120/controllers_list')
        self.init_internal_vars(self.init_pos)
        self.res = 1
        #rospy.Subscriber("/joint_states", JointState, self.joints_callback)

        # stablishes connection with simulator"
        self.gazebo = GazeboConnection(True,"SIMULATION")
        self.controllers_object = ControllersConnection("abb_irb120_3_58",self.controllers_list)
        self.gazebo.resetSimulation()

    def init_internal_vars(self, init_pos_value): # lesa me7tag yt3dl
        self.pos = init_pos_value
        self.joints = None

    #always returns the current state of the joints
    #def joints_callback(self, data): # done
        #self.joints = data

    def get_clock_time(self): # i think done 
        self.clock_time = None
        while self.clock_time is None and not rospy.is_shutdown():
            try:
                self.clock_time = rospy.wait_for_message("/clock", Clock, timeout=1.0)
                rospy.loginfo("Current clock_time READY=>" + str(self.clock_time))
            except:
                rospy.loginfo("Current clock_time not ready yet, retrying for getting Current clock_time")
        return self.clock_time
        
    def seed(self, seed=None): #overriden function # mlesh d3wa beh i think
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):#overriden function # done el mfrod
        self.STEP +=1
        rospy.loginfo(self.STEP)
        # Take actiono
        if self.res == 0:
            if action == 0: #-vejoint1
                rospy.logwarn("JOINE_1 -VE...")
                self.pos[0] -= self.pos_step
            elif action == 1: #+vejoint1
                rospy.logwarn("JOINE_1 +VE...")
                self.pos[0] += self.pos_step

            elif action == 2: #-vejoint2
                rospy.logwarn("JOINE_2 -VE...")
                self.pos[1]-= self.pos_step
            elif action == 3: #+vejoint2
                rospy.logwarn("JOINE_2 +VE...")
                self.pos[1] += self.pos_step

            elif action == 4: #-vejoint3
                rospy.logwarn("JOINE_3 -VE...")
                self.pos[2] -= self.pos_step
            elif action == 5: #+vejoint3
                rospy.logwarn("JOINE_3 +VE...")
                self.pos[2] += self.pos_step

            elif action == 6: #-vejoint4
                rospy.logwarn("JOINE_4 -VE...")
                self.pos[3] -= self.pos_step
            elif action == 7: #+vejoint4
                rospy.logwarn("JOINE_4 +VE...")
                self.pos[3] += self.pos_step

            elif action == 8: #-vejoint5
                rospy.logwarn("JOINE_5 -VE...")
                self.pos[4] -= self.pos_step
            elif action == 9: #+vejoint5
                rospy.logwarn("JOINE_5 +VE...")
                self.pos[4] += self.pos_step

            elif action == 10: #-vejoint6
                rospy.logwarn("JOINE_6 -VE...")
                self.pos[5] -= self.pos_step
            elif action == 10: #+vejoint6
                rospy.logwarn("JOINE_6 +VE...")
                self.pos[5] += self.pos_step
        else:
            self.pos= np.zeros((6,),dtype=np.float64)
            self.res = 0


        rospy.logwarn("MOVING TO POS=="+str(self.pos))

        # 1st: unpause simulation
        #rospy.loginfo("Unpause SIM...")
        self.gazebo.unpauseSim()

        self.move_joints(self.pos)
        #rospy.loginfo("Wait for some time to execute movement, time="+str(self.running_step))
        rospy.sleep(self.running_step) #wait for some time
       # rospy.loginfo("DONE Wait for some time to execute movement, time=" + str(self.running_step))

        

        # 3th: get the observation
        observation, done = self.observation_checks()

        # 4rd: pause simulation
        #rospy.loginfo("Pause SIM...")
        self.gazebo.pauseSim()

        delt = self.calculate_distance_between(self.desired_position,observation)
        # 5th: get the reward
        if not done:
            step_reward = 0
            obs_reward = self.get_reward_for_observations(observation)
            #rospy.loginfo("Reward Values: Time="+str(step_reward)+",Obs="+str(obs_reward))
            reward = step_reward + int(obs_reward)
            #rospy.loginfo("TOT Reward=" + str(reward))
        else:
            if -0.01>delt & delt<0.01 :
                reward = 1000
            else:
                reward = 1000   
        self.current_ee = observation
        now = rospy.get_rostime()

        observation = np.asarray(observation,dtype=np.float64)
        return observation, reward, done, {}

    def reset(self):


        self.STEP = 1
        self.res = 1

      #  rospy.loginfo("We UNPause the simulation to start having topic data")
        self.gazebo.unpauseSim()

        rospy.loginfo("CLOCK BEFORE RESET")
        self.get_clock_time()

       # rospy.loginfo("SETTING INITIAL POSE TO AVOID")
        self.set_init_pose()
        time.sleep(self.wait_time * 2.0)
      #  rospy.loginfo("AFTER INITPOSE CHECKING SENSOR DATA")
        #self.check_all_systems_ready()
        #rospy.loginfo("We deactivate gravity to check any reasidual effect of reseting the simulation")
        #self.gazebo.change_gravity(0.0, 0.0, 0.0)

       # rospy.loginfo("RESETING SIMULATION")
        #self.gazebo.pauseSim()
        self.gazebo.resetSim()
        self.gazebo.unpauseSim()
      #  rospy.loginfo("CLOCK AFTER RESET")
        self.get_clock_time()

       # rospy.loginfo("RESETING CONTROLLERS SO THAT IT DOESNT WAIT FOR THE CLOCK")
        #self.controllers_object.reset_controllers()
       # rospy.loginfo("AFTER RESET CHECKING SENSOR DATA")
        #self.check_all_systems_ready()
        #rospy.loginfo("CLOCK AFTER SENSORS WORKING AGAIN")
        self.get_clock_time()
        #rospy.loginfo("We reactivating gravity...")
        #self.gazebo.change_gravity(0.0, 0.0, -9.81)
        rospy.loginfo("END")

        # 7th: pauses simulation
       # rospy.loginfo("Pause SIM...")
        # get the last observation got when paused, generated by the callbakc or the check_all_systems_ready
        # Depends on who was last
        observation, done = self.observation_checks()
        observation = np.asarray(observation,dtype=np.float64)
        return observation
        
        
    '''
    UTILITY CODE FOLLOWS HERE
    '''
    
    def observation_checks(self):
        done = False
        #data = self.joints
        obs = self.get_ee_pose()     
        #state = arm.get_current_pose() # position of end effector

				
        rospy.loginfo("end effector =="+str(obs))
        position_similar = np.all(np.isclose(self.desired_position, obs, atol=1e-02))
				
        if position_similar:
            done = True
            rospy.loginfo("Reached a Desired Position!")
        
        return obs, done

    def get_reward_for_observations(self, state):
    
            position_similar = np.all(np.isclose(self.desired_position, state, atol=1e-02))
            
            new_dist_from_des_pos_ee = self.calculate_distance_between(self.desired_position,state)
            # Calculating Distance
            rospy.logwarn("desired_position="+str(self.desired_position))
            rospy.logwarn("current_pos="+str(state))
            rospy.logwarn("self.current_dist_from_des_pos_ee="+str(self.current_dist_from_des_pos_ee))
            rospy.logwarn("new_dist_from_des_pos_ee="+str(new_dist_from_des_pos_ee))
            
            delta_dist = new_dist_from_des_pos_ee - self.current_dist_from_des_pos_ee
            
            if delta_dist <0.5:
                reward = +100*(1-new_dist_from_des_pos_ee)
                #rospy.logwarn("CLOSER To Desired Position!="+str(delta_dist))
            else:
                reward = -100
                
            self.current_dist_from_des_pos_ee = new_dist_from_des_pos_ee                
            return reward
    
            
        # We update the distance
            self.current_dist_from_des_pos_ee = new_dist_from_des_pos_ee
            rospy.loginfo("Updated Distance from GOAL=="+str(self.current_dist_from_des_pos_ee))
                
            return reward


    def check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while (self._base_pub.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.loginfo("No susbribers to _base_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.loginfo("_base_pub Publisher Connected")

        while (self._pole_pub.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.loginfo("No susbribers to _pole_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass

        rospy.loginfo("All Publishers READY")

    def check_all_systems_ready(self, init=True):
        while self.base_position is None and not rospy.is_shutdown():
            try:
                self.base_position = rospy.wait_for_message("/joint_states", JointState, timeout=1.0)
                rospy.loginfo("Current /joint_states READY=>"+str(self.base_position))
                if init:
                    # We Check all the sensors are in their initial values
                    positions_ok = all(abs(i) <= 1.0e-02 for i in self.base_position.position)
                    velocity_ok = all(abs(i) <= 1.0e-02 for i in self.base_position.velocity)
                    efforts_ok = all(abs(i) <= 1.0e-01 for i in self.base_position.effort)
                    base_data_ok = positions_ok and velocity_ok and efforts_ok
                    rospy.loginfo("Checking Init Values Ok=>" + str(base_data_ok))
            except:
                rospy.logerr("Current /joint_states not ready yet, retrying for getting joint_states")
        rospy.loginfo("ALL SYSTEMS READY")


    def move_joints(self, joints_array):
        joint_value = Float64MultiArray()
        joint_value.data = joints_array
        rospy.loginfo(joints_array[0])
        #rospy.loginfo("Single Base JointsPos>>"+str(joint_value))
        self._joint_1.publish(joint_value.data[0])
        self._joint_2.publish(joint_value.data[1])
        self._joint_3.publish(joint_value.data[2])
        self._joint_4.publish(joint_value.data[3])
        self._joint_5.publish(joint_value.data[4])
        self._joint_6.publish(joint_value.data[5])

    def set_init_pose(self):
        """
        Sets joints to initial position [0,0,0]
        :return:
        """
        #self.check_publishers_connection()
        # Reset Internal pos variable
        self.init_internal_vars(self.init_pos)
        self.move_joints(self.pos)

    def get_ee_pose(self):

        grip_pos = self.arm.get_current_pose()
        gripper_pose = [grip_pos.pose.position.x, grip_pos.pose.position.y, grip_pos.pose.position.z] 
        #rospy.loginfo("EE POSE==>"+str(grip_pos))

        return gripper_pose
            
                
    def calculate_distance_between(self,v1,v2):
        """
        Calculated the Euclidian distance between two vectors given as python lists.
        """
        dist = np.linalg.norm(np.array(v1)-np.array(v2))
        return dist
