import torch.nn.functional as F
from torch import nn

from codes.model.base_model import BaseModel
from codes.model.imagination_model.util import get_num_channels_in_image


class Model(BaseModel):
    """ Convolutional encoder """

    def __init__(self, config):
        super().__init__(config=config)
        img_channels = get_num_channels_in_image(self.config.env.mode)

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 4, stride=2)
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        return x