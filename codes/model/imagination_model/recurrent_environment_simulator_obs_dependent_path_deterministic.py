import torch
import torch.nn.functional as F
from torch import nn

from codes.model.base_model import BaseModel
from codes.model.imagination_model.util import get_component
from codes.utils.util import get_product_of_iterable


class Model(BaseModel):
    """Recurrent Environment Simulator.
    This model uses the observation-dependent path"""

    def __init__(self, config):
        super(Model, self).__init__(config=config)
        self.convolutional_encoder = get_component("convolutional_encoder", config)
        self.state_transition_model = get_component("state_transition_model", config)
        self.convolutional_decoder = get_component("convolutional_decoder", config)
        self.weights = self.get_weights_dict()

    def get_weights_dict(self):
        _latent_size = self.config.model.imagination_model.latent_size
        _hidden_state_size = self.config.model.imagination_model.hidden_state_size
        _action_size = get_product_of_iterable(self.config.env.action_space["shape"])
        return torch.nn.ModuleDict({
            "w_action": torch.nn.Sequential(
                nn.Linear(_action_size, _latent_size)
            ),
            "w_h": torch.nn.Sequential(
                nn.Linear(_hidden_state_size, _latent_size)
            ),

        })

    def encode_obs(self, obs):
        obs_shape = obs.shape
        per_image_shape = obs_shape[-3:]
        batch_size = obs_shape[0]
        trajectory_length = obs_shape[1]
        num_frames = obs_shape[2]
        h_t = self.convolutional_encoder(obs.view(-1, *per_image_shape)).view(batch_size, trajectory_length, num_frames, -1)
        h_t = torch.mean(h_t, dim=2)
        return h_t, trajectory_length

    def decode_obs(self, output, trajectory_length):
        reconstructed_obs = self.convolutional_decoder(output)
        per_image_shape = reconstructed_obs.shape[-3:]
        batch_size = int(reconstructed_obs.shape[0]/trajectory_length)
        return reconstructed_obs.view(batch_size, trajectory_length, *per_image_shape)


    def forward(self, x):
        # not that x is same as x_(t-1)

        h_t, trajectory_length = self.encode_obs(obs = x.obs)
        action = x.action

        # action = (x.action).view(effetive_batch_size, -1)

        # mu, logsigma = self.variational_encoder(h_t)
        # sigma = logsigma.exp()
        # eps = torch.randn_like(sigma)
        # z_t = eps.mul(sigma).add_(mu)
        self.state_transition_model.set_state(h_t[:,0,:])
        output = []
        for t in range(0, trajectory_length):
            action_fusion = self.weights["w_action"](action[:,t,:]) * self.weights["w_h"](self.state_transition_model.h_0.squeeze(0))
            inp = torch.cat((action_fusion, h_t[:,t,:]), dim=1)
            output.append(self.state_transition_model(inp.unsqueeze(1)))
        output = torch.cat(output, dim=1).view(-1, self.config.model.imagination_model.hidden_state_size)
        reconstructed_obs = self.decode_obs(output, trajectory_length)
        return reconstructed_obs


    def loss(self, output, x):
        """ loss function """
        true_obs = x.next_obs
        # Not that we have to manually divide because of an issue in Pytorch. The fix is available only in master for now.
        return F.mse_loss(true_obs[:, :, 3, :, :, :], output) * 255 / (get_product_of_iterable(output.shape))

