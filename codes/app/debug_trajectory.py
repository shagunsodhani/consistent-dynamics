import importlib

import matplotlib.pyplot as plt
import torch

from codes.utils.argument_parser import argument_parser
from codes.utils.config import get_config
from codes.utils.util import set_seed, timing

import numpy as np
from codes.utils.util import show_tensor_as_image

@timing
def run(config_id):
    print("torch version = {}".format(torch.__version__))
    config_dict = get_config(config_id=config_id)
    set_seed(seed=config_dict.general.seed)
    module_name = "codes.data.loader.loaders"
    datatset = importlib.import_module(module_name).RolloutSequenceDataset(config=config_dict, mode="train")
    datatset.load_next_buffer()
    for idx in range(1):
        a = datatset.__getitem__(idx)[0][0]
        show_tensor_as_image((a * 255).numpy().transpose(1, 2, 0))

    # initial_state = encode_state_as_nparray(observation=observation,
    #                                         reward=reward,
    #                                         done=done,
    #                                         info=info)
    # current_trajectory.append(initial_state)
    # for action_idx in range(config_dict.dataset.num_actions_per_trajectory):
    #     current_state = torch.from_numpy(current_trajectory[-1][-1]).float()
    #     current_obs_state = torch.from_numpy(current_trajectory[-1][0]).float().view(84, 84, 3)
    #     print(current_obs_state)
    #     print(current_obs_state.shape)
    #     input("okay")


if __name__ == "__main__":
    config_id = argument_parser()
    run(config_id)
