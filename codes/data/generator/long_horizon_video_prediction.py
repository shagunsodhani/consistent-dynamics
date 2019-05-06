import os
from collections import OrderedDict

from codes.data.util import persist_trajectory, encode_state_as_nparray, persist_trajectory_as_json
from codes.envs.utils import mujoco_wrapper_using_config
from codes.model.expert_policy.utils import get_expert_policy
import torch

def from_numpy_float(state):
    if isinstance(state, (OrderedDict, dict)):
        return OrderedDict([(key, from_numpy_float(value))
            for (key, value) in state.items()])
    elif isinstance(state, (tuple, list)):
        return tuple(from_numpy_float(value) for value in state)
    else:
        return torch.from_numpy(state).unsqueeze(0).float()

def generate(config_dict):
    env = mujoco_wrapper_using_config(config_dict)
    model = get_expert_policy(config_dict)
    model.eval()

    dataset_generation_config = config_dict.dataset.dataset_generation

    path_to_persist_to = os.path.join(config_dict.dataset.base_path,
                                      config_dict.dataset.name)

    for trajectory_index in range(config_dict.dataset.num_trajectories):

        current_trajectory = []
        observation, reward, done, info = env.reset()
        observation = observation.out
        initial_state = encode_state_as_nparray(observation=observation,
                                                    reward=reward,
                                                    done=done,
                                                    info=info)
        current_trajectory.append(initial_state)
        for action_idx in range(config_dict.dataset.num_actions_per_trajectory):
            current_state = from_numpy_float(current_trajectory[-1][-1])
            action = model.sample_action(current_state)[0]
            current_trajectory.append(action)
            observation, reward, done, info = env.step(action=action)
            observation = observation.out
            current_state = encode_state_as_nparray(observation=observation,
                                                    reward=reward,
                                                    done=done,
                                                    info=info)
            current_trajectory.append(current_state)

        path_to_persist_current_trajectory = os.path.join(path_to_persist_to,
                                                          "trajectory{}".format(str(trajectory_index)))
        persist_trajectory(current_trajectory, path_to_persist_current_trajectory)

        if(dataset_generation_config.should_generate_json):
            persist_trajectory_as_json(current_trajectory, path_to_persist_current_trajectory)
