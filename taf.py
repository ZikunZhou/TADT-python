#20190513 by zikun
import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD

from taf_net import Regress_Net
from taf_rank import taf_rank_model

torch.backends.cudnn.benchmark=True

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


def taf_model(features, filter_sizes, device):
    '''
    function: select target aware feature
    args:
        filter_sizes - [batch, channel, height, width]
    '''
    num_feaure_groups = len(features)
    feature_weights = []
    channel_num = [80,300]
    nz_num = 0
    nz_num_min = 250
    balance_weights = []

    for i in range(num_feaure_groups):
        feature, filter_size = features[i], filter_sizes[i]
        reg = Regress_Net(filter_size).to(device)
        feature_size = torch.tensor(feature.shape).numpy()

        output_sigma = filter_size[-2:]* 0.1
        gauss_label = generate_gauss_label(feature_size[-2:], output_sigma).to(device)

        objective = nn.MSELoss()
        #optim = SGD(reg.parameters(),lr = 5e-7,momentum = 0.9,weight_decay = 0.0005)
        optim = SGD(reg.parameters(),lr = 1e-9,momentum = 0.9,weight_decay = 1000)
        if i == 0:
            max_epochs = 100
        else:
            max_epochs = 100
        # first train the network with mse_loss
        train_reg(reg, optim, feature, objective, gauss_label, device, max_epochs)
        reg_weights = reg.conv.weight.data

        weight = torch.sum(reg_weights, dim = (0,2,3))

        # The value ot the parameters equals to the sum of the gradients in all BP processes.
        # And we found that using the converged parameters is more stable
        sorted_cap, indices = torch.sort(torch.sum(reg_weights, dim = (0,2,3)),descending = True)

        feature_weight = torch.zeros(len(indices))
        feature_weight[indices[sorted_cap > 0]] = 1

        # we perform scale sensitive feature selection on the conv41 feaure, as it retains more spatial information
        if i == 0:
            temp_feature_weight = taf_rank_model(feature, filter_size, device)
            feature_weight = feature_weight * temp_feature_weight

        feature_weight[indices[channel_num[i]:]] = 0
        nz_num = nz_num + torch.sum(feature_weight)

        # In case，there　are two less features, we set a minmum feature number.
        # If the total number is less than the minimum number, then select more from conv4_3
        if i == 1 and nz_num < nz_num_min:
            added_indices = indices[torch.sum(feature_weight).to(torch.long): (torch.sum(feature_weight)+ nz_num_min - nz_num).to(torch.long) ]
            feature_weight[added_indices] = 1

        feature_weights.append(feature_weight)
        balance_weights.append(torch.max(torch.sum(torch.squeeze(feature)[indices[0:49],:,:],dim=0)))

    balance_weights = balance_weights[1] / torch.tensor(balance_weights, device = device)
    return feature_weights, balance_weights


def train_reg(model, optim, input, objective, gauss_label, device, epochs = 100):
    """
    function: train the regression net and regression loss
    """
    for i in range(epochs):
        input = input.to(device)
        predict = model(input).view(1,-1)

        gauss_label = gauss_label.view(1,-1)
        loss = objective(predict, gauss_label)
        if hasattr(optim,'module'):
            optim.module.zero_grad()
            loss.backward()
            optim.module.step()
        else:
            optim.zero_grad()
            loss.backward()
            optim.step()



def generate_gauss_label(size, sigma, center = (0, 0), end_pad=(0, 0)):
    """
    function: generate gauss label for L2 loss
    """
    shift_x = torch.arange(-(size[1] - 1) / 2, (size[1] + 1) / 2 + end_pad[1])
    shift_y = torch.arange(-(size[0] - 1) / 2, (size[0] + 1) / 2 + end_pad[0])

    shift_y, shift_x = torch.meshgrid(shift_y, shift_x)

    alpha = 0.2
    gauss_label = torch.exp(-1*alpha*(
                        (shift_y-center[0])**2/(sigma[0]**2) +
                        (shift_x-center[1])**2/(sigma[1]**2)
                        ))

    return gauss_label



if __name__ == '__main__':
    from feature_utils_v2 import resize_tensor

    input = torch.rand(1, 1, 3, 3)
    print(input)
    resized_input = resize_tensor(input, [7, 7])
    print(resized_input)
