import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

class Regress_Net(nn.Module):
    """
    this net exploits the ridge loss to calculate target active feature
    """
    def __init__(self, filter_size):
        """
        args:
            filter_size - [batch, channel, height, width]
        """
        super(Regress_Net, self).__init__()

        self.channels = filter_size[1]
        self.filter_height = int(2 * math.ceil(filter_size[2] / 2) + 1)
        self.filter_width = int(2 * math.ceil(filter_size[3] / 2) + 1)
        self.conv = nn.Conv2d(
            in_channels = self.channels,
            out_channels = 1,
            kernel_size = (self.filter_height, self.filter_width),
            padding = (int((self.filter_height - 1)/2), int((self.filter_width - 1)/2))
            )
        self.apply(init_weights)

    def forward(self, feature):
        return self.conv(feature)

class Rank_Net(nn.Module):
    """
    this net exploits the rank loss to calculate target activate feature
    """
    def __init__(self, filter_size):
        super(Rank_Net, self).__init__()
        self.channels = filter_size[1]
        self.filter_height = int(2 * math.ceil(filter_size[2] / 2) + 1)
        self.filter_width = int(2 * math.ceil(filter_size[3] / 2) + 1)
        self.filter_size = (self.filter_height, self.filter_width)
        self.conv = nn.Conv2d(
            in_channels = self.channels,
            out_channels = 1,
            kernel_size = self.filter_size,
            padding = 0
            )
        self.apply(init_weights)
    def forward(self, feature):
        return self.conv(feature)

def init_weights(module):
    if isinstance(module,nn.Conv2d):

        batch, channel, height, width = module.weight.data.shape
        module.weight.data = torch.randn(module.weight.data.shape)/(math.sqrt(channel*(height-1)/2*(width-1)/2)*1e8)
        nn.init.constant_(module.bias.data,0.0)




if __name__ == '__main__':
    size = torch.rand(1,512,11,11).shape
    taf = Regress_Net(size)
    #print(taf.conv.weight.data)
    #print(taf(feature))
