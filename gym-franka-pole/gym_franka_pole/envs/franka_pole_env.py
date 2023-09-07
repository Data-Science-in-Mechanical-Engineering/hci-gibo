import gym
import rospy
from gym import spaces
import numpy as np
from franka_pole.msg import CommandParameters, CommandAcceleration, CommandReset, Sample
import threading

class franka_pole_Env(gym.Env):
    def __init__(self):
        # Parameters
        self._episode_duration = 20.
        self._max_episode_steps = int(self._episode_duration * 100)
        self._length = 0.743
        self.i = 0
        self.plot_x = np.zeros(self._max_episode_steps)
        self.plot_y = np.zeros(self._max_episode_steps)
        # State
        rospy.init_node('dmp_proxy')
        self.action_space = spaces.Box(low=-1e+8, high=1e+8, shape=(3,))
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, ))
        self._real_state = np.zeros(1)
        self._real_position = np.array([0., 0.]) # Extra-state "state"
        self._real_angle = np.array([0., 0.])
        self._sim_state = np.zeros(1)
        self._sim_position = np.array([0., 0.]) # Extra-state "state"
        self._sim_angle = np.array([0., 0.])
        # Publishers & Subscribers & Locks
        self._lock = threading.Lock()
        self._real_time = 0
        self._previous_u = [0., 0.]
        self._real_condition = threading.Condition(self._lock)
        self._real_subscriber = rospy.Subscriber('/franka_pole/sample', Sample, self._real_callback)
        self._real_publisher = rospy.Publisher('/franka_pole/command_acceleration', CommandAcceleration, queue_size=10)
        self._sim_time = 0
        self._sim_condition = threading.Condition(self._lock)
        self._sim_subscriber = rospy.Subscriber('/franka_pole_sim/sample', Sample, self._sim_callback)
        self._sim_publisher = rospy.Publisher('/franka_pole_sim/command_acceleration', CommandAcceleration, queue_size=10)
        self._sim_reset = rospy.Publisher('/franka_pole_sim/command_reset', CommandReset, queue_size=10)
        self._log_tip_position_x = np.zeros(self._max_episode_steps)
        self._log_tip_position_y = np.zeros(self._max_episode_steps)
        self._log_base_position_x = np.zeros(self._max_episode_steps)
        self._log_base_position_y = np.zeros(self._max_episode_steps)
        self._log_force_x = np.zeros(self._max_episode_steps)
        self._log_force_y = np.zeros(self._max_episode_steps)
        rospy.sleep(1)

        self._reset()
    
    def _real_callback(self, sample):
        self._lock.acquire()
        self._real_time += 1
        self._real_position[0:2] = sample.franka_effector_position[0:2]
        self._real_angle[0:2] = sample.pole_angle[0:2] - np.array([-0.00475448,  0.0053806])
        self._real_angle[0] = self._real_angle[0] - 0.005 # angle_offset
        self._real_state[0] = float(self._real_time) / 100.0
        self._real_condition.notify_all()
        self._lock.release()

    def _sim_callback(self, sample):
        self._lock.acquire()
        self._sim_time += 1
        self._sim_position[0:2] = sample.franka_effector_position[0:2]
        self._sim_angle[0:2] = sample.pole_angle[0:2]
        self._sim_state[0] = float(self._sim_time) / 100.0
        self._sim_condition.notify_all()
        self._lock.release()
        
    def _transfer(self, position, angle):
        top_position = np.zeros(2)
        top_position[0] = position[0] + np.sin(angle[1]) * np.cos(angle[0]) * self._length
        top_position[1] = position[1] + np.sin(angle[0]) * np.cos(angle[1]) * self._length
        return top_position
    

    def _step(self,u):
        command = CommandAcceleration()
        command.command_effector_acceleration[0] = u[0] * 2. 
        command.command_effector_acceleration[1] = u[1] * 2.
        command.command_effector_acceleration[2] = 0.0
        _top_real_position = self._transfer(self._real_position, self._real_angle)
        _top_sim_position = self._transfer(self._sim_position, self._sim_angle)
        # logger
        self._log_tip_position_x[self.i] = _top_real_position[0]
        self._log_tip_position_y[self.i] = _top_real_position[1]
        self._log_base_position_x[self.i] = self._real_position[0]
        self._log_base_position_y[self.i] = self._real_position[1]
        self._log_force_x[self.i] = u[0] * 2.
        self._log_force_y[self.i] = u[1] * 2.
        # End of logger
        if u[-1] == 0:
            self._real_publisher.publish(command)
        else:
            self._sim_publisher.publish(command)
        scale = (2 * np.pi) / self._episode_duration
        self._lock.acquire()
        if u[-1] == 0:
            self._real_condition.wait()
            state = np.copy(self._real_state)
            cost =  np.sqrt((_top_real_position[1] - 0.12 * np.sin(scale * self._real_state[0])) ** 2 + (1.2 * (_top_real_position[0] - 0.5 - 0.2 * np.sin(scale * self._real_state[0]) * np.cos(scale * self._real_state[0]))) ** 2)
            terminated = (self._real_time > self._max_episode_steps - 1)
        else:
            self._sim_condition.wait()
            state = np.copy(self._sim_state)
            cost =  np.sqrt((_top_sim_position[1] - 0.12 * np.sin(scale * self._sim_state[0])) ** 2 + (1.2 * (_top_sim_position[0] - 0.5 - 0.2 * np.sin(scale * self._sim_state[0]) * np.cos(scale * self._sim_state[0]))) ** 2)
            terminated = (self._sim_time > self._max_episode_steps - 1)
        self._lock.release()
        self.plot_x[self.i] = _top_real_position[1] + 0.
        self.plot_y[self.i] = _top_real_position[0] - 0.5
        self.i = self.i + 1
        if terminated:
            self.i = 0
            command.command_effector_acceleration[0] = 0.0 
            command.command_effector_acceleration[1] = 0.0
            command.command_effector_acceleration[2] = 0.0
            self._sim_publisher.publish(command)
            self._real_publisher.publish(command)
            mdic = {"tx": self._log_tip_position_x, "ty": self._log_tip_position_y, "bx": self._log_base_position_x, "by": self._log_base_position_y,
            "fx": self._log_force_x, "fy": self._log_force_y}
        return state, - cost, terminated, {}

    def _reset(self):
        # Software
        command = CommandAcceleration()
        command.command_effector_acceleration = np.zeros(3)
        self._sim_publisher.publish(command)
        command = CommandReset()
        command.software = True
        self._sim_reset.publish(command)

        rospy.sleep(2)
        
        # Hardware
        command = CommandAcceleration()
        command.command_effector_acceleration = np.zeros(3)
        self._real_publisher.publish(command)
        while True:
            self._lock.acquire()
            self._real_condition.wait()
            end = (np.linalg.norm(self._real_position - np.array([0.5, 0])) < 0.01)
            self._lock.release()
            if end: break

        self._lock.acquire()
        self._real_time = 0
        self._sim_time = 0
        self._lock.release()

        return np.zeros(1)
