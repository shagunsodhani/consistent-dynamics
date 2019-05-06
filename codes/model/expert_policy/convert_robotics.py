import os
from collections import OrderedDict

import tensorflow as tf
import torch
import torch.nn as nn
import pickle

from codes.model.expert_policy.deterministic_mlp import DeterministicMLPPolicy
from codes.model.expert_policy.utils import convert_tf_variable


def convert_tf_to_pytorch(model_file):
    with open(model_file, 'rb') as f:
        data = pickle.load(f)

    graph = tf.get_default_graph()
    sess = data.sess
    state_dict = OrderedDict()
    for i in range(data.target.layers + 1):
        state_dict['layers.{0}.weight'.format(2 * i)] = convert_tf_variable(
            graph, sess, 'ddpg/target/pi/_{0}/kernel:0'.format(i)).t()
        state_dict['layers.{0}.bias'.format(2 * i)] = convert_tf_variable(
            graph, sess, 'ddpg/target/pi/_{0}/bias:0'.format(i))
    # Convert observation normalization
    state_dict['obs_rms.sum'] = convert_tf_variable(graph, sess, 'ddpg/o_stats/sum:0')
    state_dict['obs_rms.sumsq'] = convert_tf_variable(graph, sess, 'ddpg/o_stats/sumsq:0')
    count_tf = graph.get_tensor_by_name('ddpg/o_stats/count:0')
    state_dict['obs_rms.count'] = torch.tensor(sess.run(count_tf)[0], dtype=torch.float32)
    # Convert goal normalization
    state_dict['goal_rms.sum'] = convert_tf_variable(graph, sess, 'ddpg/g_stats/sum:0')
    state_dict['goal_rms.sumsq'] = convert_tf_variable(graph, sess, 'ddpg/g_stats/sumsq:0')
    count_tf = graph.get_tensor_by_name('ddpg/g_stats/count:0')
    state_dict['goal_rms.count'] = torch.tensor(sess.run(count_tf)[0], dtype=torch.float32)
    sess.close()

    # Check if the state dict can be loaded
    pi_th = DeterministicMLPPolicy(data.target.dimo, data.target.dimg, data.target.dimu,
        data.target.hidden, data.target.layers, nonlinearity=nn.ReLU, noise_eps=0.,
        random_eps=0., max_u=data.max_u, clip_obs=data.clip_obs)
    pi_th.load_state_dict(state_dict)
    del data

    return state_dict, pi_th

def main(config_dict):
    expert_policy_config = config_dict.model.expert_policy
    name = '{0}__{1}'.format(config_dict.env.name, expert_policy_config.name)
    model_file = os.path.join(expert_policy_config.save_dir, '{0}.pkl'.format(name))
    state_dict, model = convert_tf_to_pytorch(model_file)

    # Save the Pytorch model
    file_name = os.path.join(expert_policy_config.save_dir, '{0}.th.pt'.format(name))
    torch.save(state_dict, file_name)
    return model
