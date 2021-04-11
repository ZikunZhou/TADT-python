# Target-Aware Deep Tracking

Pytorch implementation of the Target-Aware Deep Tracking (TADT) method.

## Contents
- Codes of the TADT tracker.
- Codes of visualization.

## Performance

| tracker | OTB-50 | OTB2013 | OTB-100(OTB2015) |
| :-: | :-: | :-: | :-: |
|   TADT-python  | 0.615  |  \---  | 0.656 |
|[TADT-official](https://github.com/XinLi-zn/TADT) | \--- | 0.680 | 0.660 |

rate: 77FPS (i7 8700k, RTX2080)

Note: We think that the tiny performance gap between TADT-python and TADT-official is caused by the difference between Matconvnet and pytorch

## Environment
This code has been tested on Ubuntu 16.04, Python 3.7, Pytorch 1.1, CUDA 10, RTX 2080 GPU

## Installation
```
git clone git@github.com:ZikunZhou/TADT-python.git
cd TADT-python
pip install -r requirements.txt
wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat
python demo_tadt.py
```

**Note for MacOS users:** uncomment line `PyQt5` in `requirements.txt` and line `matplotlib.use('Qt5Agg')` in `tadt_tracker.py` (fixes `AttributeError: 'FigureManagerMac' object has no attribute 'window'`).

## Contact
Zikun Zhou
Email: zikunzhou@163.com
