import torch.nn.functional as F
from torch import nn

from codes.model.base_model import BaseModel
from codes.model.imagination_model.util import get_num_channels_in_image

class Model(BaseModel):
    """ Convolutional decoder """

    def __init__(self, config):
        super().__init__(config=config)
        _input_size = self.config.model.imagination_model.hidden_state_size
        img_channels = get_num_channels_in_image(self.config.env.mode)

        self.fc1 = nn.Linear(_input_size, 256)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 5, stride=2)
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
