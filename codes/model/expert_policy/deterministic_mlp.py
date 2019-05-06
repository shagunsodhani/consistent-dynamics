import numpy as np
import os
import torch
import torch.nn as nn
from torch.distributions import Distribution

from codes.model.expert_policy.utils import RunningMeanStd

class Dirac(Distribution):
    def __init__(self, value):
        super(Dirac, self).__init__()
        self.value = value

    @property
    def mean(self):
        return self.value

    def sample(self):
        return self.value

class DeterministicMLPPolicy(nn.Module):
    def __init__(self, observation_size, goal_size, output_size, hidden_size,
                 num_layers, nonlinearity=nn.ReLU, noise_eps=0.,
                 random_eps=0., max_u=1, clip_obs=200.):
        super(DeterministicMLPPolicy, self).__init__()
        self.observation_size = observation_size
        self.goal_size = goal_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.noise_eps = noise_eps
        self.random_eps = random_eps
        self.max_u = max_u
        self.clip_obs = clip_obs

        layers = []
        input_size = observation_size + goal_size
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            layers.append(nn.Linear(input_dim, hidden_size, bias=True))
            layers.append(nonlinearity())
        input_dim = input_size if num_layers == 0 else hidden_size
        layers.append(nn.Linear(input_dim, output_size, bias=True))
        self.layers = nn.Sequential(*layers)

        self.obs_rms = RunningMeanStd(shape=(observation_size,), clip_std=1e-2 ** 2, dtype=torch.float32)
        self.goal_rms = RunningMeanStd(shape=(goal_size,), clip_std=1e-2 ** 2, dtype=torch.float32)

    def forward(self, inputs):
        inputs = self._preprocess_inputs(inputs)
        action = self.layers(inputs)
        # Action post-processing
        noise = self.noise_eps * self.max_u * torch.randn(*action.shape)
        action = torch.clamp(action + noise, -self.max_u, self.max_u)
        # TODO: Handle random_eps for epsilon-greedy policies
        return Dirac(value=action)

    def _preprocess_inputs(self, inputs):
        # Normalize the inputs. Normalization happens in `ActorCritic`
        # in baselines' implementation of DDPG+HER
        observation = self.obs_rms(inputs['observation'])
        goal = self.goal_rms(inputs['desired_goal'])
        # QKFIX: We assume here that `relative_goals == False`, which is the
        # case for the policy trained on FetchPush
        observation = torch.clamp(observation, -self.clip_obs, self.clip_obs)
        goal = torch.clamp(goal, -self.clip_obs, self.clip_obs)
        return torch.cat((observation, goal), dim=1)

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

    model = DeterministicMLPPolicy(
        observation_size=int(np.prod(config_dict.env.observation_space.observation.shape)),
        goal_size=int(np.prod(config_dict.env.observation_space.goal.shape)),
        output_size=int(np.prod(config_dict.env.action_space.shape)),
        hidden_size=expert_policy_config.hidden_size,
        num_layers=expert_policy_config.num_layers,
        nonlinearity=nn.ReLU,
        noise_eps=expert_policy_config.noise_eps,
        random_eps=expert_policy_config.random_eps,
        max_u=expert_policy_config.max_u,
        clip_obs=expert_policy_config.clip_obs)

    name = '{0}__{1}'.format(config_dict.env.name, expert_policy_config.name)
    file_name = os.path.join(expert_policy_config.save_dir, '{0}.th.pt'.format(name))
    model.load_state_dict(torch.load(file_name))
    print("Model loaded successfully.")
    return model
