import gym
from gym.envs.mujoco import MujocoEnv
from gym.envs.robotics.robot_env import RobotEnv
from codes import envs
from codes.envs.normalized_env import NormalizedActionWrapper
from codes.envs.wrappers.general import FramesStack, RollAxisObservationWrapper
from codes.envs.wrappers.mujoco import (MujocoRGBObservationWrapper,
                                        MujocoGrayObservationWrapper,
                                        PusherObservationWrapper,
                                        ReacherObservationWrapper)

_MujocoEnvsTrackCamera = ['HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2',
                         'Ant-v2', 'Humanoid-v2', 'HumanoidStandup-v2']
_RoboticsEnvs = ['FetchReach-v1', 'FetchReachDense-v1',
                 'FetchPush-v1', 'FetchPushDense-v1']

def mujoco_wrapper(env_name, height=84, width=84, num_stack=4, mode='rgb', seed=42, should_render=True, get_obs=False):
    env = make_env(env_name, seed)
    if not isinstance(env.unwrapped, (MujocoEnv, RobotEnv)):
        raise ValueError('The environment `{0}` is not a MuJoCo environment.'.format(env_name))
    # The observation is an image
    kwargs = {'height': height, 'width': width, 'should_render': should_render, 'get_obs': get_obs}
    if env_name in _MujocoEnvsTrackCamera + _RoboticsEnvs:
        # If the environment is a classic Mujoco environment, set the camera to `track`
        # otherwise, set it to `external_camera_0` if it is a robotics environment
        camera_name = 'track' if (env_name in _MujocoEnvsTrackCamera) else 'external_camera_0'
        kwargs.update(camera_name=camera_name)
        # QKFIX: Robotics environments require get_obs=True
        if env_name in _RoboticsEnvs:
            kwargs['get_obs'] = True
        if mode == 'rgb':
            observation_wrapper = MujocoRGBObservationWrapper
        elif mode == 'gray':
            observation_wrapper = MujocoGrayObservationWrapper
        else:
            raise ValueError('Invalid mode `{0}` in the MuJoCo wrapper. Can be one '
                             'of [`rgb`, `gray`].'.format(mode))
    elif env_name == 'Pusher-v2':
        assert mode == 'rgb' # QKFIX: For now, Pusher and Reacher must be RGB
        observation_wrapper = PusherObservationWrapper
    elif env_name == 'Reacher-v2':
        assert mode == 'rgb' # QKFIX: For now, Pusher and Reacher must be RGB
        observation_wrapper = ReacherObservationWrapper
    else:
        raise ValueError('The environment `{0}` is not supported yet.'.format(env_name))
    env = observation_wrapper(env, **kwargs)
    # env = RollAxisObservationWrapper(env)
    if (num_stack is not None) and (num_stack > 1):
        # The observation is `num_stack` consecutive frames concatenated
        env = FramesStack(env, num_stack=num_stack)
    # Normalize the actions to be in [-1, 1]
    env = NormalizedActionWrapper(env)

    return env


def mujoco_wrapper_using_config(config_dict):
    get_obs = False
    if(config_dict.dataset.should_generate):
        # We can keep adding more usecases as required
        get_obs = True
    return mujoco_wrapper(env_name=config_dict.env.name,
                          height=config_dict.env.height,
                          width=config_dict.env.width,
                          num_stack=config_dict.env.num_stack,
                          mode=config_dict.env.mode,
                          seed=config_dict.general.seed,
                          get_obs=get_obs)


def make_env(env_name, seed=None):
    env = gym.make(env_name)
    if seed:
        env.seed(seed)
    return env
