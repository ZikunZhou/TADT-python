
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class SiameseNet(nn.Module):
    def __init__(self,feature_extractor = None, feature_weights = None):
        super(SiameseNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.feature_weights = feature_weights

    def forward(self, srch_window, exemplar):

        exemplar_size = (torch.tensor(exemplar.shape)).numpy()[-2:]
        padding = tuple((np.floor(exemplar_size/2)).astype(int))

        response = F.conv2d(srch_window, exemplar)

        output = F.pad(response, (padding[1], padding[1], padding[0], padding[0]))

        output = output/torch.max(output)
        
        assert(srch_window.shape[-2:] == output.shape[-2:]), 'Something wrong happened in doing correlation!'

        return output
