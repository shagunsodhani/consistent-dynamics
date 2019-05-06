import torch
import torch.nn.functional as F
from torch import nn

from codes.model.base_model import BaseModel
from codes.model.imagination_model.util import get_component
from codes.utils.util import get_product_of_iterable


class Model(BaseModel):
    """Recurrent Environment Simulator"""

    def __init__(self, config):
        super(Model, self).__init__(config=config)
        self.convolutional_encoder = get_component("convolutional_encoder", config)
        self.variational_encoder = get_component("variational_encoder", config)
        self.state_transition_model = get_component("state_transition_model", config)
        self.weights = self.get_weights_dict()
        # self.transition_function =

    def get_weights_dict(self):
        _latent_size = self.config.model.imagination_model.latent_size
        _action_size = get_product_of_iterable(self.config.env.action_space["shape"])
        return torch.nn.ModuleDict({
            "w_action": torch.nn.Sequential(
                nn.Linear(_action_size, _latent_size)
            ),
            "w_z": torch.nn.Sequential(
                nn.Linear(_latent_size, _latent_size)
            ),

        })

    def encode_obs(self, obs):
        obs_shape = obs.shape
        per_image_shape = obs_shape[3:]
        effetive_batch_size = get_product_of_iterable(obs_shape[:2])
        num_frames = obs_shape[2]
        h_t = self.convolutional_encoder(obs.view(-1, *per_image_shape)).view(effetive_batch_size, num_frames, -1)
        h_t = torch.mean(h_t, dim=1)
        return h_t, effetive_batch_size


    def forward(self, x):
        # not that x is same as x_(t-1)

        h_t, effetive_batch_size = self.encode_obs(obs = x.obs)

        action = (x.action).view(effetive_batch_size, -1)

        mu, logsigma = self.variational_encoder(h_t)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z_t = eps.mul(sigma).add_(mu)
        action_fusion = self.weights["w_action"](action) * self.weights["w_z"](z_t)

        # Assuming that the trajectory_length = 1
        output = self.state_transition_model((action_fusion.unsqueeze(1), h_t))
        return output


    def loss(self, output, x):
        """ loss function """
        # BCE = F.mse_loss(recon_x, x, size_average=False)
        next_obs_encoding, effetive_batch_size = self.encode_obs(x.next_obs)
        # Not that we have to manually divide because of an issue in Pytorch. The fix is available only in master for now.
        return F.mse_loss(output.squeeze(1), next_obs_encoding, reduction="none")/float(effetive_batch_size)