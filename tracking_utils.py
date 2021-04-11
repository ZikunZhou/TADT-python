import numpy as np
import torch
from scipy import signal


def fuse_feature(features):
    return torch.cat(features, dim = 1)

def visualize_res_map(res_maps, time = 20):
    num_map = res_maps.shape[0]
    fig,ax = plt.subplots(1,num_map)
    for i in range(num_map):
        res_map = res_maps[i]
        ax[i].axis('on')
        ax[i].imshow((torch.squeeze(res_map)).numpy())
    plt.ion()
    plt.pause(time)
    plt.clf()
    plt.close()


def calculate_scale(scaled_maps, scale_weights):
    '''
    args:
        scaled_maps - numlti-scale response map
        scale_weights - response map weights
    '''
    num_scale = len(scale_weights)
    assert(scaled_maps.shape[0] == num_scale), "the number of maps must equal to the number of scale_weights"
    maps = scaled_maps.view(num_scale,-1)
    max_response = torch.max(maps, dim =1)[0]
    max_response = max_response*torch.tensor(scale_weights, dtype = torch.float)
    scale_ind = torch.argmax(max_response)
    return scale_ind



def generate_2d_window(name, size, channel):
    """
    function: generate window function,such as rectangle window, hanning window, hamming window
    args:
        name -
        size - np.ndarray
        channel -
    """
    assert(name in ['hann', 'hamming']), 'window name need to be hann or hamming'
    if name == 'hann':
        h = signal.windows.hann(size[0], sym = True).reshape(size[0],1)
        w = signal.windows.hann(size[1], sym = True).reshape(1,size[1])

    else:
        h = signal.windows.hamming(size[0], sym = True).reshape(size[0],1)
        w = signal.windows.hamming(size[1], sym = True).reshape(1,size[1])
    return torch.tensor((np.tile(h*w, (channel,1,1))),dtype=torch.float)



def cal_window_size(target_size, image_size, scale_num, total_stride):
    '''
    function: calculates the size of the search window based on the input target size and the scale_num
    args:
        target_size - the size of the target
        image_size - the size of the image
        scale_num - the size ration of the search window to the target, defualt set is 3
        total_stride - the total stride of the vgg network
    returns:
        window_size - the size of window (ensure that the size the vgg16 features is odd)
        ratio - the resize ratio search_size and max_size
    '''
    search_size = target_size * scale_num
    if search_size > min(image_size):
        search_size = min(image_size)
    search_size = search_size + total_stride * 2 - (search_size - total_stride) % (total_stride * 2)
    ratio = search_size / target_size
    return search_size, ratio
