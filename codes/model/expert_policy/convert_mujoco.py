import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from baselines.common import tf_util as U
from baselines.ppo1 import mlp_policy

from codes.envs.utils import make_env
from codes.model.expert_policy.normal_mlp import NormalMLPPolicy
from codes.model.expert_policy.utils import convert_tf_variable

def convert_tf_to_pytorch(model_file, env_id, seed, num_cpu=1, hid_size=64, num_hid_layers=2):
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=hid_size, num_hid_layers=num_hid_layers)

    env = make_env(env_id, seed)
    with U.make_session(num_cpu=num_cpu) as sess:
        pi_tf = policy_fn("pi", env.observation_space, env.action_space)
        graph = tf.get_default_graph()

        # Load the TensorFlow model
        saver = tf.train.Saver(pi_tf.get_variables())
        saver.restore(sess, model_file)

        # Convert layers
        state_dict = OrderedDict()
        for i in range(num_hid_layers):
            state_dict['layers.{0}.weight'.format(2 * i)] = convert_tf_variable(
                graph, sess, 'pi/pol/fc{0}/kernel:0'.format(i + 1)).t()
            state_dict['layers.{0}.bias'.format(2 * i)] = convert_tf_variable(
                graph, sess, 'pi/pol/fc{0}/bias:0'.format(i + 1))
        # Convert last layer
        state_dict['mean.weight'] = convert_tf_variable(graph, sess, 'pi/pol/final/kernel:0').t()
        state_dict['mean.bias'] = convert_tf_variable(graph, sess, 'pi/pol/final/bias:0')
        # Convert log std
        state_dict['logstd'] = convert_tf_variable(graph, sess, 'pi/pol/logstd:0')[0]
        # Convert observation normalization
        state_dict['obs_rms.sum'] = convert_tf_variable(graph, sess, 'pi/obfilter/runningsum:0')
        state_dict['obs_rms.sumsq'] = convert_tf_variable(graph, sess, 'pi/obfilter/runningsumsq:0')
        count_tf = graph.get_tensor_by_name('pi/obfilter/count:0')
        state_dict['obs_rms.count'] = torch.tensor(sess.run(count_tf), dtype=torch.float64)

    # Check if the state dict can be loaded
    pi_th = NormalMLPPolicy(int(np.prod(env.observation_space.shape)), int(np.prod(env.action_space.shape)),
        hid_size, num_hid_layers, nonlinearity=nn.Tanh)
    pi_th.load_state_dict(state_dict)

    return state_dict, pi_th

def main(config_dict):
    expert_policy_config = config_dict.model.expert_policy
    name = '{0}__{1}'.format(config_dict.env.name, expert_policy_config.name)
    model_file = os.path.join(expert_policy_config.save_dir, '{0}.ckpt'.format(name))
    state_dict, model = convert_tf_to_pytorch(model_file, config_dict.env.name,
        config_dict.general.seed,
        num_cpu=expert_policy_config.num_cpu,
        hid_size=expert_policy_config.hidden_size,
        num_hid_layers=expert_policy_config.num_layers)

    # Save the Pytorch model
    file_name = os.path.join(expert_policy_config.save_dir, '{0}.th.pt'.format(name))
    torch.save(state_dict, file_name)
    return model

def test_conversion(config_dict):
    expert_policy_config = config_dict.model.expert_policy
    name = '{0}__{1}'.format(config_dict.env.name, expert_policy_config.name)
    model_file_tf = os.path.join(expert_policy_config.save_dir, '{0}.ckpt'.format(name))
    model_file_th = os.path.join(expert_policy_config.save_dir, '{0}.th.pt'.format(name))

    env = make_env(config_dict.env.name, config_dict.general.seed)


    pi_tf = mlp_policy.MlpPolicy(name='pi', ob_space=env.observation_space,
        ac_space=env.action_space, hid_size=expert_policy_config.hidden_size,
        num_hid_layers=expert_policy_config.num_layers)
    observations_tf = []
    with U.make_session(num_cpu=expert_policy_config.num_cpu) as sess:
        # Load TF model
        saver = tf.train.Saver(pi_tf.get_variables())
        saver.restore(tf.get_default_session(), model_file_tf)
        # Sample trajectory
        # env.seed(config_dict.general.seed)
        observation, done = env.reset(), False
        observations_tf.append(observation)
        while not done:
            action = pi_tf.act(stochastic=False, ob=observation)[0]
            observation, _, done, _ = env.step(action)
            observations_tf.append(observation)

    pi_th = NormalMLPPolicy(int(np.prod(env.observation_space.shape)),
        int(np.prod(env.action_space.shape)), expert_policy_config.hidden_size,
        expert_policy_config.num_layers, nonlinearity=nn.Tanh)
    observations_th = []
    # Load Pytorch model
    with open(model_file_th, 'rb') as f:
        state_dict = torch.load(f)
        pi_th.load_state_dict(state_dict)
    # Sample trajectory
    env.seed(config_dict.general.seed)
    observation, done = env.reset(), False
    observations_th.append(observation)
    while not done:
        observation_tensor = torch.from_numpy(observation).unsqueeze(0).float()
        action_tensor = pi_th(observation_tensor).mean[0]
        action = action_tensor.detach().cpu().numpy()
        observation, _, done, _ = env.step(action)
        observations_th.append(observation)

    # Compare the trajectories
    linf_norm = np.max(np.abs(np.asarray(observations_tf) - np.asarray(observations_th)))
    print('Maximum absolute difference between observations: {0}'.format(linf_norm))
