import torch
from path import Path
from torch._C import device

from dataloader import DataLoaderImgFile
from eval import evaluate
from net import WordDetectorNet
from visualization import visualize_and_plot
import cv2
import numpy as np

def ceil32(val):
    if val % 32 == 0:
        return val
    val = (val // 32 + 1) * 32
    return val

def get_scale_factor(f):
    return f if f < 1 else 1

def infer(imgpath, max_side_len=1024, device='cuda'):

    net = WordDetectorNet()
    net.load_state_dict(torch.load('../model/weights', map_location=device))
    net.eval()
    net.to(device)

    orig = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

    f = min(max_side_len / orig.shape[0], max_side_len / orig.shape[0])
    if f < 1:
        orig = cv2.resize(orig, dsize=None, fx=f, fy=f)
    img = np.ones((ceil32(orig.shape[0]), ceil32(orig.shape[1])), np.uint8) * 255
    img[:orig.shape[0], :orig.shape[1]] = orig

    img = (img / 255 - 0.5).astype(np.float32)
    imgs = img[None, None, ...]

    res, boxes = evaluate(imgs, net, device, max_aabbs=1000)

    f = get_scale_factor(f)
    aabbs = [aabb.scale(1 / f, 1 / f) for aabb in boxes]
    visualize_and_plot(res, aabbs)