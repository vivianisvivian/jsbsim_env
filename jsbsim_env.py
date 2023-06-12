import gym
import numpy as np
from my_jsbsim_env.simulation import Simulation
from my_jsbsim_env.catalogs.catalog import Catalog as c
import random
import math

# 归一化示例，建议写到训练侧代码中
# def normalize(state):
#     normal = np.array([100, 10, 10,
#                         1, 0.1,
#                         1, 1,
#                         100, 1, 1, 
#                         0.1, 0.1, 0.1, 60])
#     s_dim = len(state)
#     return state/normal[:s_dim]

class JSBSimEnv(gym.Env):

    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    An JsbSimEnv is instantiated with a Task that implements a specific
    aircraft control task with its own specific observation/action space and
    variables and agent_reward calculation.

    ATTRIBUTION: this class implements the OpenAI Gym Env API. Method
    docstrings have been adapted or copied from the OpenAI Gym source code.
    """

    metadata = {"render.modes": ["human", "csv"]}

    def __init__(self, task):
        """

        Constructor. Init some internal state, but JSBSimEnv.reset() must be

        called first before interacting with environment.

        :param task: the Task for the task agent is to perform

        """

        self.sim = None
        self.task = task()
        self.observation_space = self.task.get_observation_space()  # None
        self.action_space = self.task.get_action_space()  # None

        self.state = None

    def step(self, action=None):
        """

        Run one timestep of the environment's dynamics. When end of

        episode is reached, you are responsible for calling `reset()`

        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).



        :param action: np.array, the agent's action, with same length as action variables.

        :return:

            state: agent's observation of the current environment

            reward: amount of reward returned after previous action

            done: whether the episode has ended, in which case further step() calls are undefined

            info: auxiliary information

        """
        info = " "

        if not len(action) == len(self.action_space.spaces):
            raise ValueError("mismatch between action:{} and action space {}".format(len(action), len(self.action_space.spaces)))

        if len(action) == 4:   # 对油门进行反归一化
            action[-1] = ((action[-1] + 1)/2) * 0.9

        self.state = self.make_step(action)

        reward = self.task.get_reward(self.state, self.sim)

        done, info = self.is_terminal()

        state = self.state if not done else self._get_clipped_state()  # returned state should be in observation_space

        s = np.asarray(state).T.reshape(-1)
        
        return s, reward, done, info

    def make_step(self, action=None):
        """

        Calculates new state.


        :param action: array of floats, the agent's last action

        :return: observation: array, agent's observation of the environment state


        """
        # take actions
        if action is not None:
            self.sim.set_property_values(self.task.get_action_var(), action)

        # run simulation
        self.sim.run()

        return self.get_observation()

    def reset(self, init_s=None, random_init=None, random_aircraft=None):
        """
        init_s, RIS : 一个trick, 先不用，从中间采样起始位置
        Resets the state of the environment and returns an initial observation.
        :return: array, the initial observation of the space.  注意：这里也需要归一化
        """
        if self.sim:
            self.sim.close()
        else:  # 固定起始高度，固定飞机型号
            self.sim = Simulation(
                aircraft_name=self.task.aircraft_name,
                init_conditions=self.task.init_conditions,
                jsbsim_freq=self.task.jsbsim_freq,
                agent_interaction_steps=self.task.agent_interaction_steps,
            )
        self.state = self.get_observation()

        self.observation_space = self.task.get_observation_space()

        self.action_space = self.task.get_action_space()

        s = np.asarray(self.state).T.reshape(-1)
        return s

    def is_terminal(self):
        """

        Checks if the state is terminal.

        :return: bool

        """

        # is_not_contained = not self.observation_space.contains(self.state)

        # return is_not_contained or self.task.is_terminal(self.state, self.sim)
        return self.task.is_terminal(self.state, self.sim)

    def render(self, mode="human", **kwargs):
        """Renders the environment.

        The set of supported modes varies per environment. (And some

        environments do not support rendering at all.) By convention,

        if mode is:

        - human: print on the terminal
        - csv: output to cvs files

        Note:

            Make sure that your class's metadata 'render.modes' key includes

              the list of supported modes. It's recommended to call super()

              in implementations to use the functionality of this method.



        :param mode: str, the mode to render with
        """
        return self.task.render(self.sim, mode=mode, **kwargs)


    def close(self):
        """Cleans up this environment's objects
        Environments automatically close() when garbage collected or when the
        program exits.

        """
        if self.sim:
            self.sim.close()

    def get_observation(self):
        """
        get state observation from sim.
        :return: NamedTuple, the first state observation of the episode

        """
        obs_list = self.sim.get_property_values(self.task.get_observation_var())
        return tuple([np.array([obs]) for obs in obs_list])

    def get_sim_time(self):
        """ Gets the simulation time from sim, a float. """
        return self.sim.get_sim_time()

    def get_state(self):
        return self.sim.get_sim_state()

    def _get_clipped_state(self):
        clipped = [
            np.clip(self.state[i], o.low, o.high) if self.task.state_var[i].clipped else self.state[i]
            for i, o in enumerate(self.observation_space)
        ]
        return tuple(clipped)

    def set_state(self, state):
        self.sim.set_sim_state(state)
        self.state = self.get_observation()