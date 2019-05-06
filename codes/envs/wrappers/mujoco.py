import numpy as np
import gym
from gym import spaces
from codes.utils.util import show_tensor_as_image
from gym.envs.mujoco import PusherEnv, ReacherEnv

class MujocoRGBObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, height=84, width=84, should_render=False,
                 get_obs=False, camera_name='track'):
        super(MujocoRGBObservationWrapper, self).__init__(env)
        self.height = height
        self.width = width
        self.should_render = should_render
        self.camera_name = camera_name
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(height, width, 3), dtype=np.uint8)
        if not get_obs:
            self.env.unwrapped._get_obs = lambda: None

    def viewer_setup(self):
        if self.camera_name not in self.env.unwrapped.sim.model.camera_names:
            raise ValueError('Camera with name `{0}` was not found in '
                '`{1}`.'.format(self.camera_name, self.env.unwrapped))
        camera_id = self.env.unwrapped.sim.model.camera_name2id(self.camera_name)
        self.env.unwrapped.viewer.cam.type = 2
        self.env.unwrapped.viewer.cam.fixedcamid = camera_id
        # Hide the overlay
        self.env.unwrapped.viewer._hide_overlay = True

    def observation(self, observation):
        # Observation of the form HXWXC
        if 'rgb_array' not in self.metadata['render.modes']:
            raise AttributeError()
        if hasattr(self.env.unwrapped, '_render_callback'):
            self.env.unwrapped._render_callback()
        if self.env.unwrapped.viewer is None:
            self.env.unwrapped._get_viewer()
            self.viewer_setup()
        viewer = self.env.unwrapped._get_viewer()
        if(self.should_render):
            viewer.render()

        observation = viewer.read_pixels(self.width, self.height, depth=False)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # Add the full state to infos
        info.update(state=self.env.unwrapped._get_obs())
        return self.observation(observation), reward, done, info

    def reset(self):

        state = self.env.reset()
        rewards = None
        done = None
        info = {
            "reward_run": None,
            "reward_ctrl": None,
            "state": state
        }
        return self.observation(state), rewards, done, info

class MujocoGrayObservationWrapper(MujocoRGBObservationWrapper):
    def __init__(self, env, height=84, width=84, should_render=True,
                 get_obs=False, camera_name='track'):
        super(MujocoGrayObservationWrapper, self).__init__(env,
            height=height, width=width, should_render=should_render,
            get_obs=get_obs, camera_name=camera_name)
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(height, width, 1), dtype=np.uint8)

    def observation(self, observation):
        observation = super(MujocoGrayObservationWrapper,
            self).observation(observation)
        # QKFIX: Use raw Numpy instead of OpenCV for conversion RGB2Gray to
        # avoid any conflict with mujoco_py. The weightings are given by
        # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
        observation = np.dot(observation, [0.299, 0.587, 0.114])
        observation = np.expand_dims(observation, axis=2)

        return observation

class PusherObservationWrapper(MujocoRGBObservationWrapper):
    def __init__(self, env, height=84, width=84, should_render=True,
                 get_obs=False):
        super(PusherObservationWrapper, self).__init__(env,
            height=height, width=width, should_render=should_render,
            get_obs=get_obs, camera_name=None)
        assert isinstance(self.env.unwrapped, PusherEnv)

    def viewer_setup(self):
        body_id = self.env.unwrapped.model.geom_name2id('table')
        self.env.unwrapped.viewer.cam.type = 0
        lookat = self.env.unwrapped.sim.data.geom_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.env.unwrapped.viewer.cam.lookat[idx] = value
        self.env.unwrapped.viewer.cam.distance = 3
        self.env.unwrapped.viewer.cam.azimuth = 0
        self.env.unwrapped.viewer.cam.elevation = -90.
        # Hide the overlay
        self.env.unwrapped.viewer._hide_overlay = True

class ReacherObservationWrapper(MujocoRGBObservationWrapper):
    def __init__(self, env, height=84, width=84, should_render=True,
                 get_obs=False):
        super(ReacherObservationWrapper, self).__init__(env,
            height=height, width=width, should_render=should_render,
            get_obs=get_obs, camera_name=None)
        assert isinstance(self.env.unwrapped, ReacherEnv)

    def viewer_setup(self):
        body_id = self.env.unwrapped.model.geom_name2id('ground')
        self.env.unwrapped.viewer.cam.type = 0
        lookat = self.env.unwrapped.sim.data.geom_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.env.unwrapped.viewer.cam.lookat[idx] = value
        self.env.unwrapped.viewer.cam.distance = 0.68
        self.env.unwrapped.viewer.cam.azimuth = 0
        self.env.unwrapped.viewer.cam.elevation = -90.
        # Hide the overlay
        self.env.unwrapped.viewer._hide_overlay = True
