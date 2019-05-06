import importlib
import os
import random
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from attr import attrs, attrib

from codes.utils.log import write_message_logs


class BaseModel(nn.Module):
    '''Base class for all models'''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = "base_model"
        self.description = "This is the base class for all the models. All the other models should " \
                      "extend this class. It is not to be used directly."
        self.criteria = nn.CrossEntropyLoss()
        self.similarity_criteria = nn.PairwiseDistance(p=2)

    def loss(self, outputs, labels):
        '''Method to perform the loss computation'''
        return self.criteria(outputs, labels)

    def track_loss(self, outputs, labels):
        '''There are two different functions related to loss as we might be interested in
         tracking one loss and optimising another'''
        return self.loss(outputs, labels)

    def save_model(self, epochs=-1, optimizers=None, is_best_model=False):
        '''Method to persist the model'''
        model_config = self.config.model
        state = {
            "epochs": epochs + 1,
            "state_dict": self.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "np_random_state": np.random.get_state(),
            "python_random_state": random.getstate(),
            "pytorch_random_state": torch.get_rng_state()
        }
        if is_best_model:
            path = os.path.join(model_config.save_dir,
                                "best_model.tar")
        else:
            path = os.path.join(model_config.save_dir,
                                "model_epoch_" + str(epochs + 1) + "_timestamp_" + str(int(time())) + ".tar")
        torch.save(state, path)
        write_message_logs("saved model to path = {}".format(path))

    def load_model(self, optimizers):
        '''Method to load the model'''
        model_config = self.config.model
        path = model_config.load_path
        write_message_logs("Loading model from path {}".format(path))
        if (self.config.device == "cuda"):
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        epochs = checkpoint["epochs"]
        self._load_metadata(checkpoint)
        self._load_model_params(checkpoint["state_dict"])
        for optim_index, optimizer in enumerate(optimizers):
            # optimizer.load_state_dict(checkpoint[OPTIMIZERS][optim_index]())
            optimizer.load_state_dict(checkpoint["optimizers"][optim_index])
        return optimizers, epochs

    def _load_metadata(self, checkpoint):
        np.random.set_state(checkpoint["np_random_state"])
        random.setstate(checkpoint["python_random_state"])
        torch.set_rng_state(checkpoint["pytorch_random_state"])

    def _load_model_params(self, state_dict):
        self.load_state_dict(state_dict)

    def get_model_params(self):
        model_parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        write_message_logs("Total number of params = " + str(params))
        return model_parameters

    def get_optimizers_and_schedulers(self):
        '''Method to return the list of optimizers and schedulers for the model'''
        optimizers = self.get_optimizers()
        if optimizers:
            optimizers, schedulers = self._register_optimizers_to_schedulers(optimizers)
            return optimizers, schedulers
        return None

    def get_optimizers(self):
        '''Method to return the list of optimizers for the model'''
        optimizers = []
        model_params = self.get_model_params()
        if (model_params):
            optimizers.append(self._register_params_to_optimizer(model_params))
            return optimizers
        return None

    def _register_params_to_optimizer(self, model_params):
        # Method to map params to an optimizer
        optimizer_config = self.config.model.optimizer
        optimizer_cls = getattr(importlib.import_module("torch.optim"), optimizer_config.name)
        return optimizer_cls(
                model_params,
                lr=optimizer_config.learning_rate,
                weight_decay=optimizer_config.l2_penalty
            )

    def _register_optimizers_to_schedulers(self, optimizers):
        # Method to map optimzers to schedulers
        optimizer_config = self.config.model.optimizer
        if (optimizer_config.scheduler_type == "exp"):
            schedulers = list(map(lambda optimizer: optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=self.config.model.optimizer.scheduler_gamma), optimizers))
        elif (optimizer_config.scheduler_type == "plateau"):
            schedulers = list(map(lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode="min", patience=self.config.model.optimizer.scheduler_patience,
                factor=self.config.model.optimizer.scheduler_gamma, verbose=True), optimizers))

        return optimizers, schedulers

    def forward(self, data):
        '''Forward pass of the network'''
        pass

    def get_param_count(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False
