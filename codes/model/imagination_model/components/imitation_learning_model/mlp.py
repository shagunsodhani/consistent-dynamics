from torch import nn

from codes.model.base_model import BaseModel
from codes.utils.util import get_product_of_iterable


class Model(BaseModel):
    """ Imitation Model."""

    def __init__(self, config):
        super().__init__(config=config)
        self._input_size = self.config.model.imagination_model.hidden_state_size + self.config.model.imagination_model.latent_size
        # Note that the hidden state size corresponds to the encoding of the observation in the pixel space.
        self._output_size = get_product_of_iterable(self.config.env.action_space.shape)
        self.policy = nn.Sequential(
            nn.Linear(self._input_size, self._input_size),
            nn.ReLU(),
            nn.Linear(self._input_size, int(self._input_size / 2)),
            nn.ReLU(),
            nn.Linear(int(self._input_size / 2), int(self._input_size / 4)),
            nn.ReLU(),
            nn.Linear(int(self._input_size / 4), self._output_size)
        )
        self.criteria = nn.MSELoss()

    def forward(self, x):
        action = self.policy(x)
        return action
