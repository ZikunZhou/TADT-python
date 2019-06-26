#tadt-demo
import glob
from os.path import join, realpath, dirname
import numpy as np
import scipy.io
from tadt_tracker import Tadt_Tracker
from backbone_v2 import build_vgg16

def load_sequece(root_path):
    img_list = (glob.glob(join(root_path, '*/img/*.jpg')))
    img_list.sort()
    gt_path = glob.glob(join(root_path, '*/*.txt'))

    with open(gt_path[0], 'r') as f:
        gt_bboxes = f.readlines()
    if '\t' in gt_bboxes[0]:
        spl = '\t'
    else:
        spl = ','
    gt_bboxes = np.array([list(map(int,gt_bbox.strip('\n').split(spl))) for gt_bbox in gt_bboxes]).astype(int)
    return img_list, gt_bboxes

if __name__ == "__main__":
    from defaults import _C as cfg
    import time
    import torch
    assert(False), 'please download "imagenet-vgg-verydeep-16.mat" from "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat" and set its path in defaults.py'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_path = join(realpath(dirname(__file__)),'sequences/')
    img_list, gt_bboxes = load_sequece(root_path)

    #------------------demo------------------------------------------------------------------
    model = build_vgg16(cfg)
    tracker = Tadt_Tracker(cfg, model = model, device = device, display = True)
    tracker.initialize_tadt(img_list[0], gt_bboxes[0])
    #if want to visualize the selected feature, uncomment these lines
    #tracker.visualize_feature(
    #                        features = tracker.features,
    #                        stage = 'conv4_3',
    #                        srch_window_size = (180,180),
    #                        subwindow = tracker.subwindow,
    #                        feature_weights = tracker.feature_weights,
    #                        balance_weights = tracker.balance_weights
    #                        )
    for i in range(1, len(img_list)):
       tracker.tracking(img_list[i], i)

    print('fps: ',tracker.cal_fps(i))
    #name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    #with open(root_path+name+'.txt','w') as f:
    #    for bbox in results:
    #        newline = str(bbox) + '\n'
    #        f.write(newline)
