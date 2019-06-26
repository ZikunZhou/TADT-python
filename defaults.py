# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from os.path import join, realpath, dirname
from yacs.config import CfgNode as CN
#YACS was created as a lightweight library to define and manage system configurations

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.MAX_SIZE = 59
_C.MODEL.MIN_SIZE = 44
_C.MODEL.SCALE_NUM = 3
if _C.MODEL.SCALE_NUM ==3:
    # as the feature map size is 45×45，we use the change step of 2,
    # which benefits center position alignment.
    _C.MODEL.SCALES = [45/47, 1, 45/43]
    _C.MODEL.SCALE_WEIGHTS = [0.99, 1, 1.0055]
elif _C.MODEL.SCALE_NUM ==5:
    _C.MODEL.SCALES = [1-4/45, 1-2/45, 1, 1+2/45, 1+4/45]
    _C.MODEL.SCALE_WEIGHTS = [0.985, 0.988, 1, 1.005, 1.006]
_C.MODEL.TOTAL_STRIDE = 4

_C.BACKBONE = CN()
_C.BACKBONE.VGG16 = CN()
_C.BACKBONE.VGG16.DEPTH = 16
_C.BACKBONE.VGG16.WITH_BN = False
_C.BACKBONE.VGG16.WITH_POOLS = (True,True,False,False)
_C.BACKBONE.VGG16.NUM_STAGES = 4
_C.BACKBONE.VGG16.DILATIONS = (1,1,1,1)
_C.BACKBONE.VGG16.FROZEN_STAGE = 4
_C.BACKBONE.VGG16.BN_EVAL = False
_C.BACKBONE.VGG16.BN_FROZEN = False
_C.BACKBONE.VGG16.CEIL_MODEL = False
_C.BACKBONE.VGG16.OUT_INDICES = ['conv4_1', 'conv4_3']
_C.BACKBONE.VGG16.PRETRAIN_MAT = join(realpath(dirname(__file__)),'imagenet-vgg-verydeep-16.mat')
