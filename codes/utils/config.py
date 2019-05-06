import datetime
import json
import os
from copy import deepcopy
from time import time
# from gym import make

import yaml
from addict import Dict

from codes.utils.util import make_dir, get_current_commit_id


def _read_config(config_id=None):
    '''
    Method to read the config file and return as a dict
    '''
    path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]
    config_name = "config.yaml"
    if (config_id):
        config_name = "{}.yaml".format(config_id)
    return yaml.load(open(os.path.join(path, 'config', config_name)))


def _read_sample_config():
    '''
    Method to read the config file and return as a dict
    :return:
    '''
    path = os.path.abspath(os.pardir).split('/codes')[0]
    return yaml.load(open(os.path.join(path, 'config', 'sample.config.yaml')))


def _get_boolean_value(value):
    if (type(value) == bool):
        return value
    elif (value.lower() == "true"):
        return True
    else:
        return False


def get_config(config_id=None, should_make_dir = True):
    '''Method to prepare the config for all downstream tasks'''

    config = get_base_config(config_id)

    config = _post_process(config, should_make_dir)

    if is_valid_config(config, config_id):
        return config

    else:
        return None

def is_valid_config(config, config_id):
    '''Simple tests to check the validity of a given config file'''
    if config.general.id == config_id:
        return True
    else:
        print("Error in Config. Config Id and Config Names do not match")
        return False

def get_sample_config():
    '''Method to prepare the config for all downstream tasks'''

    config = get_sample_base_config()

    config = _post_process(config, should_make_dir=False)

    return config


def _post_process(config, should_make_dir):
    # Post Processing on the config addict

    config.general = _post_process_general_config(deepcopy(config.general))
    config.env = _post_process_env_config(deepcopy(config.env))
    config.dataset = _post_process_dataset_config(deepcopy(config.dataset), config.general, should_make_dir)
    config.model = _post_process_model_config(deepcopy(config.model), config, should_make_dir)
    config.log = _post_process_log_config(deepcopy(config.log), config.general, should_make_dir)
    config.plot = _post_process_plot_config(deepcopy(config.plot), config.general, should_make_dir)
    # if(config.general.server=="cluster"):
    #     config = _update_config_using_env(config)

    return config


def _post_process_general_config(general_config):
    # Method to post process the general section of the config

    if "seed" not in general_config:
        general_config.seed = 42
    else:
        general_config.seed = int(general_config.seed)

    if ("base_path" not in general_config) or (general_config.base_path == ""):
        general_config.base_path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]

    if ("device" not in general_config) or (general_config.device == ""):
        general_config.device = "cpu"

    if ("id" not in general_config) or (general_config.id == ""):
        general_config.id = str(int(time()))
    else:
        general_config.id = str(general_config.id)

    if ("commit_id" not in general_config) or (general_config.commit_id == ""):
        general_config.commit_id = get_current_commit_id()

    if ("server" not in general_config) or (general_config.server == ""):
        general_config.env = "slurm"

    if ("date" not in general_config) or (not general_config.date):
        now = datetime.datetime.now()
        general_config.date = now.strftime("%Y-%m-%d %H:%M")

    return general_config


def _post_process_env_config(env_config):
    # Method to post process the env section of the config
    default_params = {
        "name": "HalfCheetah-v2",
        "height": 84,
        "width": 84,
        "num_stack": 1,
        "mode": "rgb",
    }

    for key in default_params:
        if key not in env_config:
            env_config[key] = default_params[key]

    env_config.observation_space = _post_process_observation_space_config(deepcopy(env_config.observation_space))
    env_config.action_space = _post_process_action_space_config(deepcopy(env_config.action_space))

    return env_config

def _post_process_observation_space_config(observation_space_config):
    default_params =  {
        "shape": [17,]
    }

    observation_space_config = _copy_from_default_dict(observation_space_config, default_params)
    return observation_space_config

def _post_process_action_space_config(action_space_config):
    default_params = {
        "shape": [6,]
    }

    return _copy_from_default_dict(action_space_config, default_params)

def _post_process_dataset_config(dataset_config, general_config, should_make_dir):
    # Method to post process the dataset section of the config
    if ("base_path" not in dataset_config) or (dataset_config.base_path == ""):
        dataset_config.base_path = os.path.join(general_config.base_path, "data")

    if(should_make_dir):
        make_dir(dataset_config.base_path)

    if ("name" not in dataset_config) or (dataset_config.name == ""):
        dataset_config.name = "HalfCheetah-v2"

    if(should_make_dir):
        make_dir(os.path.join(dataset_config.base_path, dataset_config.name))

    default_params = {
        "should_generate": False,
        "num_trajectories": 10000,
        "num_actions_per_trajectory": 1,
        "batch_size": 100,
        "buffer_size": 100,
        "num_workers": 2,
        "should_load": True,
        "sequence_length": 250,
        "imagination_length": 10,
    }

    dataset_config = _copy_from_default_dict(dataset_config, default_params)

    for key in ["should_generate"]:
        if key in dataset_config:
            dataset_config[key] = _get_boolean_value(dataset_config[key])

    dataset_config.split = _post_process_split_config(deepcopy(dataset_config.split))

    dataset_config.dataset_generation = _post_process_dataset_generation_config(
        deepcopy(dataset_config.dataset_generation))

    return dataset_config


def _post_process_split_config(split_config):
    default_params = {
        "train": 0.8,
        "val": 0.05,
        "test": 0.15
    }
    return _copy_from_default_dict(split_config, default_params)

def _post_process_expert_policy_config(expert_policy_config, general_config, should_make_dir):
    # Method to post process the expert_policy section of the config
    default_params = {
        "name": "ppo",
        "num_timesteps": 1000000,
        "save_dir": "",
        "hidden_size": 64,
        "num_layers": 2,
        "num_cpu": 1,
        "should_convert_model": True
    }
    expert_policy_config = _copy_from_default_dict(expert_policy_config, default_params)

    if (expert_policy_config.save_dir == ""):
        expert_policy_config.save_dir = os.path.join(general_config.base_path,
                                                     "model",
                                                     "expert_policy")
    elif (expert_policy_config.save_dir[0] != "/"):
        expert_policy_config.save_dir = os.path.join(general_config.base_path,
                                                     "model",
                                                     "expert_policy")

    if(should_make_dir):
        make_dir(path=expert_policy_config.save_dir)

    return expert_policy_config



def _post_process_dataset_generation_config(dataset_generation_confg):
    # Method to post process the net section of the config

    default_params = {
        "dataset_id": "long_horizon_video_prediction",
        "should_generate_json": False,
    }

    return _copy_from_default_dict(dataset_generation_confg, default_params)


def _post_process_model_config(model_config, config, should_make_dir):
    # Method to post process the model section of the config

    general_config = config.general

    default_params = {
        "name": "baseline1",
        "should_train": False,
        "num_epochs": 1000,
        "persist_per_epoch": -1,
        "persist_best_model": False,
        "early_stopping_patience": 1,
        "save_dir": "",
        "should_load_model": False,
        "optimizer": Dict(),
        "imagination_model": Dict(),
        "expert_policy":Dict(),
        "modes": ["train", "val", "test"],
    }

    for key in default_params:
        if key not in model_config:
            model_config[key] = default_params[key]

    if (model_config.save_dir == ""):
        model_config.save_dir = os.path.join(general_config.base_path, "model", general_config.id)
    elif (model_config.save_dir[0] != "/"):
        model_config.save_dir = os.path.join(general_config.base_path, model_config.save_dir)

    if(should_make_dir):
        make_dir(path=model_config.save_dir)

    model_config.load_path = os.path.join(general_config.base_path,
                                          "model", model_config.load_path)

    for key in ["should_load_model"]:
        model_config[key] = _get_boolean_value(model_config[key])

    model_config.optimizer = _post_process_optimizer_config(deepcopy(model_config.optimizer))
    model_config.imagination_model = _post_process_imagination_model_config(deepcopy(model_config.imagination_model))
    model_config.expert_policy = _post_process_expert_policy_config(deepcopy(model_config.expert_policy),
                                                                    general_config, should_make_dir)

    return model_config


def _post_process_imagination_model_config(imagination_model_config):
    default_params = {
        "name": "learning_to_query",
        "latent_size": 100,
        "hidden_state_size": 1024,
        "consistency_model": Dict({
            "name": "euclidean",
            "alpha": 1.0
        }),
        "imitation_learning_model":Dict({
            "name": "mlp",
            "should_train": False,
            "alpha": 1.0
        })
    }

    imagination_model_config = _copy_from_default_dict(imagination_model_config, default_params)
    return imagination_model_config




def _post_process_optimizer_config(optimizer_config):
    # Method to post process the optimizer section of the config

    default_params = {
        "name": "Adam",
        "learning_rate": 0.0001,
        "scheduler_type": "exp",
        "scheduler_gamma": 1.0,
        "scheduler_patience": 10,
        "l2_penalty": 0
    }

    for key in default_params:
        if key not in optimizer_config:
            optimizer_config[key] = default_params[key]

    return optimizer_config


def _post_process_plot_config(plot_config, general_config, should_make_dir):
    # Method to post process the plot section of the config
    if ("base_path" not in plot_config) or (plot_config.base_path == ""):
        plot_config.base_path = os.path.join(general_config.base_path,
                                             "plots", general_config.id)
        if(should_make_dir):
            make_dir(path=plot_config.base_path)

    return plot_config


def _post_process_log_config(log_config, general_config, should_make_dir):
    # Method to post process the log section of the config

    if ("file_path" not in log_config) or (log_config.file_path == ""):
        log_config.file_path = os.path.join(general_config.base_path,
                                            "logs", general_config.id)
        if(should_make_dir):
            make_dir(path=log_config.file_path)
        log_config.file_path = os.path.join(log_config.file_path, "log.txt")

    log_config.dir = log_config.file_path.rsplit("/", 1)[0]
    return log_config


def get_base_config(config_id=None):
    # Returns the bare minimum config (addict) needed to run the experiment
    config_dict = _read_config(config_id)
    return Dict(config_dict)


def get_sample_base_config():
    # Reads the sample config and returns as addict
    config_dict = _read_sample_config()
    return Dict(config_dict)


def _copy_from_default_dict(final_dict, default_dict):
    for key in default_dict:
        if key not in final_dict:
            final_dict[key] = default_dict[key]
    return final_dict

def _update_config_using_env(config):
    from codes.envs.utils import make_env
    env = make_env(config.env.name)
    config.env.observation_space.shape = env.observation_space.shape
    config.env.action_space.shape = env.action_space.shape
    del env
    return config

if __name__ == "__main__":
    print(json.dumps(get_config()))
