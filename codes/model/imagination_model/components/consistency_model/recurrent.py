import torch
from torch import nn
import numpy as np

from codes.model.base_model import BaseModel
from codes.model.imagination_model.util import merge_first_and_second_dim


class Model(BaseModel):
    """ Model to differentiate between the two input distributions """

    def __init__(self, config):
        super().__init__(config=config)
        self._hidden_state_size = self.config.model.imagination_model.hidden_state_size
        self._input_size = self._hidden_state_size
        # Note that the hidden state size corresponds to the encoding of the observation in the pixel space.
        self._num_layers = 1
        self._batch_first = True
        self.h_0 = None
        self.c_0 = None
        self.reset_state()
        self._is_bidirectional = True
        self.lstm = nn.LSTM(input_size=self._input_size,
                            hidden_size=self._hidden_state_size,
                            num_layers=self._num_layers,
                            batch_first=self._batch_first,
                            bidirectional=self._is_bidirectional)
        self.mlp = nn.Sequential(
            nn.Linear(self._input_size, self._input_size),
            nn.ReLU(),
            nn.Linear(self._input_size, self._input_size),
            nn.ReLU(),
            nn.Linear(self._input_size, self._input_size)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self._input_size, 1),
        )

        self.criteria = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, x):
        return self.prepare_data_and_forward(x)

    def prepare_data_and_forward(self, x):
        '''This function first prepares the data for the discriminator and then feeds the data to the disriminator.
        Note that we want to encapsulate this data processing aspect into the discriminator itself and hence
        a call to forward function would call this function.
        We have this function to make it explicit that we are doing both data processing and forward pass'''
        self.reset_state()
        open_loop_dist, close_loop_dist = x
        open_loop_lstm_encoding = self._get_lstm_encoding(open_loop_dist)
        open_loop_lstm_encoding = self.mlp(merge_first_and_second_dim(open_loop_lstm_encoding))
        open_loop_true_output = torch.ones(open_loop_lstm_encoding.shape[0], 1)

        close_loop_lstm_encoding = self._get_lstm_encoding(close_loop_dist)
        close_loop_lstm_encoding = self.mlp(merge_first_and_second_dim(close_loop_lstm_encoding))
        close_loop_true_output = torch.zeros(close_loop_lstm_encoding.shape[0], 1)


        lstm_encoding = torch.cat(
            (open_loop_lstm_encoding, close_loop_lstm_encoding), dim=0
        )

        true_output = torch.cat(
            (open_loop_true_output, close_loop_true_output), dim=0
        )

        permutation = np.random.permutation(true_output.shape[0])

        true_output = (true_output[permutation]).to(lstm_encoding.device)
        lstm_encoding = lstm_encoding[permutation]

        predicted_output = self.classifier(lstm_encoding).to(lstm_encoding.device)

        loss_discriminator = self.criteria(predicted_output, true_output)
        loss_close_loop = loss_discriminator*(1-true_output)
        return torch.mean(loss_close_loop).unsqueeze(0), torch.mean(loss_discriminator).unsqueeze(0)

    def reset_state(self):
        self.h_0 = None
        self.c_0 = None

    def set_state(self, h_0):
        self.h_0 = h_0.unsqueeze(0).contiguous()
        self.c_0 = torch.zeros_like(self.h_0)

    def _get_lstm_encoding(self, data):
        '''Pass the data through a bidirectional LSTM and average the embeddings corresponding to the two directions.'''
        return torch.mean(
            torch.cat(
                tuple(
                    map(
                        lambda _tensor: _tensor.unsqueeze(0), torch.chunk(self.lstm(data)[0], 2, dim=2))),
                dim=0),
            dim=0)
