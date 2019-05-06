import importlib
import torch

from codes.experiment import experiment
from codes.utils.argument_parser import argument_parser
from codes.utils.config import get_config
from codes.utils.util import set_seed, timing, show_tensor_as_image
from codes.experiment.bootstrap import setup_model
from codes.utils.log import set_logger, write_message_logs, write_config_log
import time

torch.set_num_threads(2)

@timing
def run(config_id):
    config_dict = bootstrap(config_id)
    if (config_dict.dataset.should_generate):
        generate_dataset(config_dict)
    # if (config_dict.dataset.should_load):
    #     load_dataset(config_dict)
    if (config_dict.model.should_train):
        train_model(config_dict)
    if config_dict.model.expert_policy.should_convert_model:
        module_name = "codes.app.convert_model"
        importlib.import_module(module_name).convert_model_from_tf_to_th(config_dict)


def bootstrap(config_id):
    config_dict = get_config(config_id=config_id)
    print(config_dict.log)
    set_logger(config_dict)
    write_message_logs("Starting Experiment at {}".format(time.asctime(time.localtime(time.time()))))
    write_message_logs("torch version = {}".format(torch.__version__))
    write_config_log(config_dict)
    set_seed(seed=config_dict.general.seed)
    return config_dict


def generate_dataset(config_dict):
    module_name = "codes.data.generator." + config_dict.dataset.dataset_generation.dataset_id
    importlib.import_module(module_name).generate(config_dict)


def load_dataset(config_dict):
    module_name = "codes.data.loader.loaders"
    dataset = importlib.import_module(module_name).RolloutSequenceDataset(config=config_dict, mode="train")
    dataset.load_next_buffer()
    a = dataset.__getitem__(1)[0][0]
    # show_tensor_as_image((a[3]*255).numpy().transpose(1, 2, 0))
    model, _, _, _ = setup_model(config_dict)
    model.eval()
    model()


def train_model(config_dict):
    if (config_dict.model.name == "expert_policy"):
        if config_dict.model.expert_policy.name == "normal_mlp":
            from codes.model.expert_policy.train_mujoco import main as train_ppo
            train_ppo(config_dict)
    else:
        experiment.run_experiment(config_dict)


if __name__ == "__main__":
    config_id = argument_parser()
    run(config_id)
