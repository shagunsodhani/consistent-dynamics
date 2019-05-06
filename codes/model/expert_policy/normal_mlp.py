import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from codes.model.expert_policy.utils import RunningMeanStd


class NormalMLPPolicy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers,
                 nonlinearity=nn.Tanh):
        super(NormalMLPPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity

        layers = []
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            layers.append(nn.Linear(input_dim, hidden_size, bias=True))
            layers.append(nonlinearity())
        self.layers = nn.Sequential(*layers)

        self.mean = nn.Linear(hidden_size, output_size, bias=True)
        self.logstd = nn.Parameter(torch.zeros(output_size))
        self.obs_rms = RunningMeanStd(shape=(input_size,), dtype=torch.float64)

    def forward(self, inputs):
        normalized_inputs = self.obs_rms(inputs)
        normalized_inputs = torch.clamp(normalized_inputs, -5.0, 5.0)
        outputs = self.layers(normalized_inputs)
        mean = self.mean(outputs)
        return Normal(loc=mean, scale=torch.exp(self.logstd))

    def load_weights(self, config_dict):
        expert_policy_config = config_dict.model.expert_policy
        name = '{0}__{1}'.format(config_dict.env.name, expert_policy_config.name)

        # Load the Pytorch model
        with open(os.path.join(expert_policy_config.save_dir, '{0}.pt'.format(name)), 'wb') as f:
            self.load_state_dict(torch.load())

    def sample_action(self, input):
        return self.forward(input).mean.detach().numpy()



def get_model_using_config_dict(config_dict):
    expert_policy_config = config_dict.model.expert_policy

    model = NormalMLPPolicy(input_size=int(np.prod(config_dict.env.observation_space.shape)),
                            output_size=int(np.prod(config_dict.env.action_space.shape)),
                            hidden_size=expert_policy_config.hidden_size,
                            num_layers=expert_policy_config.num_layers,
                            nonlinearity=nn.Tanh)

    name = '{0}__{1}'.format(config_dict.env.name, expert_policy_config.name)
    file_name = os.path.join(expert_policy_config.save_dir, '{0}.th.pt'.format(name))
    model.load_state_dict(torch.load(file_name))
    print("Model loaded successfully.")
    return model
