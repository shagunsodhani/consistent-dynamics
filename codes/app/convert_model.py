from codes.model.expert_policy.convert_mujoco import main as convert_mujoco
from codes.model.expert_policy.convert_robotics import main as convert_robotics


def convert_model_from_tf_to_th(config_dict):
    if config_dict.model.expert_policy.name == 'deterministic_mlp':
        convert_robotics(config_dict)
    elif config_dict.model.expert_policy.name == 'normal_mlp':
        convert_mujoco(config_dict)
    else:
        raise NotImplementedError('Unknown expert policy with '
            'name `{0}`'.format(config_dict.model.expert_policy.name))
