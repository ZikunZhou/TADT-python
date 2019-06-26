import sys
sys.path.append('/home/zikun/work/tracking/pysot-toolkit')
from pysot.datasets import DatasetFactory
from pysot.utils.region import vot_overlap
import numpy as np
from tadt_tracker import Tadt_Tracker
import os
import torch

from defaults import _C as cfg
from backbone_v2 import build_vgg16


otb_root = 'the root path of otb benchmark'
result_path = 'the root path that you want to put the result'
vgg16_model_mat_path = 'the path of the vgg model'
otb_root = '/home/zikun/work/data/OTB2015-python/'
result_path = '/home/zikun/Desktop/tadt_release/results/TADT_otb100/'
vgg16_model_mat_path = "/home/zikun/Desktop/tadt_release/imagenet-vgg-verydeep-16.mat"
dataset = DatasetFactory.create_dataset(name = 'OTB100', dataset_root = otb_root, load_img = False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
frame_counter = 0
pred_bboxes = []
model = build_vgg16(cfg)


if not os.path.exists(result_path):
    os.mkdir(result_path)
for video in dataset:
    #if video.name != 'CarScale':
    #    continue
    print('video',video.name)
    for idx, (img_path, gt_bbox) in enumerate(video):
        print('frame:',idx)
        if idx == frame_counter:
            target_location = np.array(gt_bbox)
            tracker = Tadt_Tracker(cfg, model = model, device = device, display = False)
            tracker.initialize_tadt(img_path, target_location)
        else:
            tracker.tracking(img_path, idx)
    tracker.end_visualize()
    tracker.saving_result(result_path, video.name, zero_index = False)
