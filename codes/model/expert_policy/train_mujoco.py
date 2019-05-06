import tensorflow as tf
import os
# For some reason, I need to have this import statement otherwise my code crashes
import mujoco_py

from baselines.common.cmd_util import make_mujoco_env, make_robotics_env
from baselines.common import tf_util as U
from baselines import logger

from codes.envs.utils import _MujocoEnvsTrackCamera, _RoboticsEnvs
_MujocoEnvs = _MujocoEnvsTrackCamera + ['Pusher-v2', 'Reacher-v2']

def train(env_id, num_timesteps, seed, hid_size=64, num_hid_layers=2):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    assert env_id in (_MujocoEnvs + _RoboticsEnvs)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=hid_size, num_hid_layers=num_hid_layers)
    if env_id in _MujocoEnvs:
        env = make_mujoco_env(env_id, seed)
    elif env_id in _RoboticsEnvs:
        env = make_robotics_env(env_id, seed)
    else:
        raise ValueError('Environment `{0}` is not supported.'.format(env_id))
    # Not putting these params in config as we do not plan on changing them.
    optim_epochs = 10 if env_id in _MujocoEnvs else 5
    optim_batchsize = 64 if env_id in _MujocoEnvs else 256
    pi = pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=optim_epochs, optim_stepsize=3e-4,
            optim_batchsize=optim_batchsize,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()
    return pi

def save(pi, sess, name):
    saver = tf.train.Saver(pi.get_variables())
    save_dir = os.path.join(logger.get_dir(), name)
    saver.save(sess, save_dir)

def main(config_dict):
    expert_policy_config = config_dict.model.expert_policy
    logger.configure(dir=expert_policy_config.save_dir)
    with U.make_session(num_cpu=expert_policy_config.num_cpu) as sess:
        pi = train(config_dict.env.name,
                   num_timesteps=expert_policy_config.num_timesteps,
                   seed=config_dict.general.seed,
                   hid_size=expert_policy_config.hidden_size,
                   num_hid_layers=expert_policy_config.num_layers)
        name = '{0}__{1}.ckpt'.format(config_dict.env.name, expert_policy_config.name)
        save(pi, sess, name)