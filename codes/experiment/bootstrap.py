import importlib

import torch
from addict import Dict

from codes.data.dataloader import get_dataloaders
from codes.data.datastructure import get_model_data_spec_from_config
from codes.metric.metric_registry import get_trackable_metric_dict


def setup_model(config):
    model_type_name = config.model.name
    model_name = config.model[model_type_name].name
    module_name = ".".join(["codes", "model", model_type_name, model_name])
    model_cls = getattr(importlib.import_module(module_name), "Model")
    model = model_cls(config)
    optimizers, schedulers = model.get_optimizers_and_schedulers()
    epochs = 0
    if (config.model.should_load_model):
        optimizers, epochs = model.load_model(optimizers)

    return model, optimizers, schedulers, epochs



def _setup_experiment(config):
    experiment = Dict()
    experiment.config = config
    experiment.support_modes = config.model.modes
    experiment.dataloaders = get_dataloaders(config, experiment.support_modes)
    experiment.device = torch.device(config.general.device)
    experiment.model, experiment.optimizers, experiment.schedulers, experiment.epoch_index = setup_model(
        config)
    experiment.model = experiment.model.to(experiment.device)

    experiment.validation_metrics_dict = get_trackable_metric_dict(time_span=config.model.early_stopping_patience)
    experiment.metric_to_perform_early_stopping = "loss"
    experiment.data_container_cls = get_model_data_spec_from_config(config)["container"]
    return experiment

