import torch
import numpy as np
import torch.nn as nn
from torch.optim import SGD
import math

from taf_net import Rank_Net
from feature_utils_v2 import resize_tensor
from rank_loss import RankLoss
import matplotlib.pyplot as plt


torch.backends.cudnn.benchmark=True

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

def taf_rank_model(feature, filter_size, device):
    rank_net = Rank_Net(filter_size).to(device)
    rank_loss = RankLoss()
    filter_size = np.append(filter_size[0:2], np.array(rank_net.filter_size))
    temp_feature_weight = rank_selection(feature, filter_size, rank_net, rank_loss, device)
    return temp_feature_weight

def rank_selection(feature, filter_size, model, loss, device):
    '''
    funtion: selects the scale sensitive features based on the ranking loss
    args:
        filter_size - [batch, channel, height, width]
    '''
    feature.requires_grad_()
    scale_samples, pair_labels = generate_ranked_samples(feature, filter_size[-2:])
    scale_samples = scale_samples.to(device)
    pair_labels = pair_labels.to(device)
    #----------------------------------------------------------------------------
    #optim = SGD(model.parameters(),lr = 1e-9,momentum = 0.9,weight_decay = 1000)
    #rank_train(model, optim, scale_samples, loss, pair_labels, epochs=0)
    #rank_weights = model.conv.weight.data
    #sorted_rank_cap, rank_indices = torch.sort(torch.sum(rank_weights, dim = (0,2,3)),descending = True)
    #----------------------------------------------------------------------------
    #gradients = rank_eval(scale_samples, pair_labels, model, loss)
    rank_eval(scale_samples, pair_labels, model, loss)
    gradients = feature.grad
    sorted_rank_cap, rank_indices = torch.sort(torch.sum(gradients, dim = (0,2,3)),descending = True)
    feature.detach_()
    #----------------------------------------------------------------------------
    temp_weight = torch.zeros(len(rank_indices))
    temp_weight[rank_indices[sorted_rank_cap > 0]] = 1
    return temp_weight

def rank_eval(scale_samples, pair_labels, model, loss):
    """
    funtion: backward
    args:
        scale_samples - shape num_of_scales×１×height×width
        pair_labels - shape 32×２
        model - rank_net
        loss - rank_loss
    """
    #scale_samples.requires_grad_()
    predicts = torch.squeeze(model(scale_samples))
    loss_lsep = loss(predicts, pair_labels)
    #---------------------------------------------------------------------
    pre_grads = loss.backward(torch.tensor(1))
    model.zero_grad()
    predicts.backward(pre_grads)
    #---------------------------------------------------------------------
    return scale_samples.grad

def generate_ranked_samples(feature, filter_size):
    """
    function: generates samples with different scales and offsets based on the input target positions
              make the size ratio from 0.5 to 2 as possible
    args:
        feature -
        filter_size - [height, width]
    results:
        samples -
        pair_labels -
    """
    feature_size = torch.tensor(feature.shape).numpy()#[batch, channel, height, width
    assert(np.prod((filter_size % 2).astype(int)) == 1), 'filter_size need to be an odd number.\n'
    # target_location in feature [y_c, x_c, height, width]
    target_location = np.append(np.round(feature_size[-2:]/2-1), filter_size)
    b_filter_size = np.floor(filter_size / 2).astype(int)#[height, width]
    feature_c_height = (np.floor(feature_size[-2] / 2) + 1).astype(int)
    re_sizes = (np.arange(np.max(b_filter_size), 2*(feature_c_height - 1) + 1) * 2 + 1).astype(int)
    c_index = (np.where(re_sizes == feature_size[-2]))[0][0]

    left_pad_num = c_index#0-index
    right_pad_num = feature_size[-2] - (c_index + 1)
    pad_num = min(left_pad_num,right_pad_num)
    re_sizes = re_sizes[c_index - pad_num: c_index + pad_num + 1]
    ratios = re_sizes / feature_size[-2]
    target_locations = np.concatenate((
            (-b_filter_size[0] + np.floor((re_sizes-1)/2))[:,np.newaxis].astype(int),
            (-b_filter_size[1] + np.floor((re_sizes-1)/2))[:,np.newaxis].astype(int),
            (b_filter_size[0] + np.floor((re_sizes-1)/2))[:,np.newaxis].astype(int),
            (b_filter_size[1] + np.floor((re_sizes-1)/2))[:,np.newaxis].astype(int)
            ), axis = 1
    )

    re_features = [resize_tensor(feature, (re_size,re_size), align_corners = True) for re_size in re_sizes]
    target_features = [re_feat[:,:,loc[0]:loc[2]+1, loc[1]:loc[3]+1] for (loc, re_feat) in zip(target_locations,re_features)]
    samples = torch.cat(target_features, dim = 0)

    labels = 1 - (ratios - 1) ** 2
    pair_labels = generate_pair_label(labels)
    return samples, pair_labels

def generate_pair_label(labels):
    """
    function: converts the labels of a set of scale labels (from the smallest candidate to the largest one)
              into pair-wise labels
    """
    num_label = len(labels)
    base_index = math.ceil(num_label / 2)-1 #0-index
    pair_label = []
    for i in range(1, base_index):
        pair_label.append(torch.tensor(list(zip(
            list(range(i, base_index+1)) + list(range(base_index, num_label-i)),
            list(range(0, base_index+1-i)) + list(range(base_index+i, num_label))
            )), dtype = torch.long))
    #print(pair_label)
    return torch.cat(pair_label, dim = 0)

def rank_train(model,optim, input, objective, pair_labels, epochs):
    """
    function: train the rank net and rank loss
    """
    input.requires_grad_()
    for i in range(epochs):
        predicts = torch.squeeze(model(input))
        loss_lsep = objective(predicts, pair_labels)

        #pre_grads = objective.backward(torch.tensor(1))
        if hasattr(optim,'module'):
            optim.module.zero_grad()
            loss_lsep.backward()
            optim.module.step()
        else:
            optim.zero_grad()
            loss_lsep.backward()
            optim.step()

def tensor_show(tensor, time = 20,bbox = None, normalize = True, feature = False, scale = 1):

    fig,ax = plt.subplots(1,1)
    if normalize and not feature:
        ax.imshow(unnormalize(tensor))
    elif not normalize and not feature:
        ax.imshow(torch.squeeze(tensor).to('cpu').numpy().transpose((1,2,0)))
    elif feature:
        ax.imshow(torch.squeeze(tensor).to('cpu').numpy())
    ax.text(0, 0, 'frame: {}'.format(scale), fontsize=15)
    if bbox is not None:
        ax.add_patch(Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],fill=False,color='g'))
    plt.savefig("{}.png".format(scale))
    plt.ion()
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(100,100,800,500)
    plt.pause(time)
    plt.clf()
    plt.close()
