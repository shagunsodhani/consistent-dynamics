import numpy as np
import json
from addict import Dict
from collections import OrderedDict

def persist_trajectory(trajectory, path_to_persist_to):
    '''Method to persist the trajectory'''
    np.save(path_to_persist_to, np.asarray(trajectory))

def persist_trajectory_as_json(trajectory, path_to_persist_to):
    '''Method to persist the trajectory'''
    file_name = path_to_persist_to+".json"
    with open(file_name, "w") as f:
        decoded_trajectory = _decode_trajectory_from_nparray_as_list(trajectory)
        f.write(json.dumps(decoded_trajectory))
        f.write("\n")

def encode_state_as_nparray(observation, reward=None, done=None, info={}):
    '''Method to encode a given state'''
    info = info if info is not None else {}
    state = OrderedDict()
    state["observation"] = observation
    state["reward"] = reward
    state["done"] = done
    state["reward_run"] = info["reward_run"] if 'reward_run' in info else None
    state["reward_ctrl"] = info["reward_ctrl"] if 'reward_ctrl' in info else None
    state["state"] = info["state"] if 'state' in info else None
    return np.asarray(list(state.values()))

def decode_state_from_nparray(encoded_state):
    '''Method to decode a given numpy state'''
    decoded_state = Dict()
    decoded_state.observation = encoded_state[0]
    decoded_state.reward = encoded_state[1]
    decoded_state.done = encoded_state[2]
    decoded_state.info.reward_run = encoded_state[3]
    decoded_state.info.reward_ctrl = encoded_state[4]
    decoded_state.info.state = encoded_state[5]
    return [decoded_state.observation,
            decoded_state.reward,
            decoded_state.done,
            decoded_state.info]

def decode_trajectory_from_nparray(trajectory):
    '''Method to decode a given numpy trajectory'''
    decoded_trajectory = []
    decoded_trajectory.append(decode_state_from_nparray(trajectory[0]))
    current_idx = 1
    while current_idx<len(trajectory):
        current_decoded_action = trajectory[current_idx].tolist()
        decoded_trajectory.append(current_decoded_action)
        current_idx+=1
        current_decoded_state = decode_state_from_nparray(trajectory[current_idx])
        decoded_trajectory.append(current_decoded_state)
        current_idx+=1
    return decoded_trajectory

def _decode_trajectory_from_nparray_as_list(trajectory):
    '''Method to decode a given numpy trajectory into a python list based construct.
    The difference between this method and decode_trajectory_from_nparray method is this method does not use any
    numpy constructs. This method is written to make debuggin easier and should not be needed otherwise.'''
    decoded_trajectory = decode_trajectory_from_nparray(trajectory)
    for idx in range(0, len(decoded_trajectory), 2):
        decoded_trajectory[idx] = decoded_trajectory[idx][0].tolist()
    return decoded_trajectory