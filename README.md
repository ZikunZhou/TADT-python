# Target-Aware Deep Tracking

Pytorch implementation of the Target-Aware Deep Tracking (TADT) method.

## Main contents:
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

## Requirements
numpy, cv2, matplotlib, scipy, yacs


## Installation
1. Clone the GIT repository:  
$ git clone    
2. Run the demo script to test the tracker:  
python demo_tadt.py


## Contact
Zikun Zhou
Email: zikunzhou@163.com
