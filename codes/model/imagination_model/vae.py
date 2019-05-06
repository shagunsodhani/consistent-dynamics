# WE DONOT USE THIS CODE ANYMORE
# Note that most of this code is modified from https://github.com/ctallec/world-models/blob/1420916db9eb3e40963f9e67dce1225b1b28efd2/models/vae.py

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image
import os

from codes.model.base_model import BaseModel
from codes.model.imagination_model.util import get_num_channels_in_image


class Model(BaseModel):
    """ Variational Autoencoder """

    def __init__(self, config):
        super(Model, self).__init__(config=config)
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma

    def loss(self, recon_x, x, mu, logsigma):
        """ VAE loss function """
        # BCE = F.mse_loss(recon_x, x, size_average=False)
        BCE = F.mse_loss(recon_x, x, size_average=True)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
        return BCE + KLD

    def sample_from_model(self, epoch):
        with torch.no_grad():
            device = torch.device(self.config.general.device)
            latent_size = self.config.model.imagination_model.latent_size
            height = self.config.env.height
            sample = torch.randn(1, latent_size).to(device)
            sample = self.decoder(sample).cpu()
            save_image(sample.view(1, 3, height, height),
                       os.path.join(self.config.model.save_dir, 'sample_' + str(epoch) + '.png'))

class Decoder(BaseModel):
    """ VAE decoder """

    def __init__(self, config):
        super(Decoder, self).__init__(config=config)
        latent_size = self.config.model.imagination_model.latent_size
        img_channels = get_num_channels_in_image(self.config.env.mode)

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = F.sigmoid(self.deconv4(x))
        return reconstruction


class Encoder(BaseModel):
    """ VAE encoder """

    def __init__(self, config):
        super(Encoder, self).__init__(config=config)
        latent_size = self.config.model.imagination_model.latent_size
        img_channels = get_num_channels_in_image(self.config.env.mode)

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc_mu = nn.Linear(2 * 2 * 256, latent_size)
        self.fc_logsigma = nn.Linear(2 * 2 * 256, latent_size)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma
