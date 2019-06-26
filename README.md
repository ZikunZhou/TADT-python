# Target-Aware Deep Tracking

Pytorch implementation of the Target-Aware Deep Tracking (TADT) method.

## Main contents:
- Codes of the TADT tracker.
- Codes of visualization.

## Performance

| tracker | OTB-50 | OTB-100 |
| ------- | ------ | ------- |
|   TADT  | 0.615  | 0.656   |

rate: 65FPS

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
