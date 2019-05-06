import torch
import torch.nn as nn
import importlib

class RunningMeanStd(nn.Module):
    def __init__(self, epsilon=1e-2, shape=(), clip_range=None,
                 clip_std=1e-2, dtype=torch.float64):
        super(RunningMeanStd, self).__init__()
        self.epsilon = epsilon
        self.shape = shape
        self.clip_range = clip_range
        self.clip_std = clip_std
        self.dtype = dtype

        self.register_buffer('sum', torch.zeros(shape, dtype=dtype))
        self.register_buffer('sumsq', torch.full(shape, epsilon, dtype=dtype))
        self.register_buffer('count', torch.tensor(epsilon, dtype=dtype))

    @property
    def mean(self):
        mean = self.sum / self.count
        return mean.float()

    @property
    def std(self):
        second_moment = self.sumsq / self.count
        var = torch.max(second_moment.float() - self.mean ** 2, torch.tensor(self.clip_std))
        return torch.sqrt(var)

    def forward(self, inputs):
        with torch.no_grad():
            normalized = (inputs - self.mean) / self.std
            if self.clip_range is not None:
                return torch.clamp(normalized, -self.clip_range, self.clip_range)
            else:
                return normalized

    def update(self, inputs):
        self.sum += torch.sum(inputs, dim=0).to(self.dtype)
        self.sumsq += torch.sum(inputs ** 2, dim=0).to(self.dtype)
        self.count += inputs.size(0)

def get_expert_policy(config_dict):
    module_name = "codes.model.expert_policy.{}".format(config_dict.model.expert_policy.name)
    return importlib.import_module(module_name).get_model_using_config_dict(config_dict)

def convert_tf_variable(graph, sess, name):
    variable = graph.get_tensor_by_name(name)
    return torch.from_numpy(sess.run(variable))
