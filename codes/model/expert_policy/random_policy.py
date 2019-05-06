import torch
import torch.nn as nn


class RandomPolicy(nn.Module):
    def __init__(self, config_dict):
        super(RandomPolicy, self).__init__()
        self.output_size = config_dict.model.expert_policy.num_allowed_actions + 1

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        batch_size = 1
        return 2 * torch.rand((batch_size, 6)) - 1

    def sample_action(self, input):
        return self.forward(input).numpy()


def get_model_using_config_dict(config_dict):
    return RandomPolicy(config_dict)
