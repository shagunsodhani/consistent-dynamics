from gym.envs.registration import register
import gym

# Mujoco
# ----------------------------------------


def setup_envs():
    env_names = ['Reacher-v2', 'Pusher-v2', 'Thrower-v2', 'Striker-v2',
        'InvertedPendulum-v2', 'InvertedDoublePendulum-v2', 'HalfCheetah-v2',
        'Hopper-v2', 'Swimmer-v2', 'Walker2d-v2', 'Ant-v2', 'Humanoid-v2',
        'HumanoidStandup-v2']

    for env_name in env_names:
        name, version = env_name.split('-', 1)
        register_once(
            id='{0}PixelRGB-v0'.format(name),
            entry_point='codes.envs.utils:mujoco_wrapper',
            kwargs={'env_name': env_name, 'height': 64, 'width': 64, 'num_stack': 1,
                    'mode': 'rgb', 'should_render': True, 'get_obs': False}
        )

        register_once(
            id='{0}PixelGray-v0'.format(name),
            entry_point='codes.envs.utils:mujoco_wrapper',
            kwargs={'env_name': env_name, 'height': 64, 'width': 64, 'num_stack': 1,
                    'mode': 'gray', 'should_render': True, 'get_obs': False}
        )

        register_once(
            id='{0}PixelRGBDebug-v0'.format(name),
            entry_point='codes.envs.utils:mujoco_wrapper',
            kwargs={'env_name': env_name, 'height': 64, 'width': 64, 'num_stack': 1,
                    'mode': 'rgb', 'should_render': True, 'get_obs': True}
        )

        register_once(
            id='{0}PixelGrayDebug-v0'.format(name),
            entry_point='codes.envs.utils:mujoco_wrapper',
            kwargs={'env_name': env_name, 'height': 64, 'width': 64, 'num_stack': 1,
                    'mode': 'gray', 'should_render': True, 'get_obs': True}
        )

def register_once(id, entry_point, kwargs):
    if id not in gym.envs.registry.env_specs:
        register(
            id=id,
            entry_point=entry_point,
            kwargs=kwargs
        )



setup_envs()
