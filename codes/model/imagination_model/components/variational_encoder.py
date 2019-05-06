from torch import nn

from codes.model.base_model import BaseModel
from codes.model.imagination_model.util import get_action_space_size


class Model(BaseModel):
    """ Variational Encoder """

    def __init__(self, config):
        super().__init__(config=config)
        _latent_size = self.config.model.imagination_model.latent_size
        _input_size = self._get_input_size()
        self.fc_mu = nn.Linear(_input_size, _latent_size)
        self.fc_logsigma = nn.Linear(_input_size, _latent_size)

    def forward(self, x):
        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)
        return mu, logsigma

    def _get_input_size(self):
        _action_size = get_action_space_size(self.config)
        _input_size = self.config.model.imagination_model.hidden_state_size + _action_size
        return _input_size
