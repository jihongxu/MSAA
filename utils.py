import random
import math
from PIL import Image
import numpy as np
import torch
import argparse
import json
from datetime import datetime
from datetime import date
from uuid import UUID


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, date):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, type(bytes)):
            return str(obj, encoding='utf-8')
        elif isinstance(obj, type(np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, UUID):
            return obj.hex
        else:
            return json.JSONEncoder.default(self, obj)


"""
图片数据增强的自定义Transform, 作用是随机选取图像上一个区域，
并将该区域随机填充为灰色或白色, 这样做可以增加图像数据的多样性，防止过拟合。 

random erasing其实和cutout非常类似，也是一种模拟物体遮挡情况的数据增强方法。区别在于，
cutout是把图片中随机抽中的矩形区域的像素值置为0，相当于裁剪掉，random  erasing是用随机数或者数据集中像素的平均值
替换原来的像素值。而且，cutout每次裁剪掉的区域大小是固定的，Random erasing替换掉的区域大小是随机的。 
"""


class RandomErase(object):
    def __init__(self, prob, sl, sh, r):
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r = r

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob:
            return img

        while True:
            area = random.uniform(self.sl, self.sh) * img.size[0] * img.size[1]
            ratio = random.uniform(self.r, 1 / self.r)

            h = int(round(math.sqrt(area * ratio)))
            w = int(round(math.sqrt(area / ratio)))

            if h < img.size[0] and w < img.size[1]:
                x = random.randint(0, img.size[0] - h)
                y = random.randint(0, img.size[1] - w)
                img = np.array(img)
                if len(img.shape) == 3:
                    for c in range(img.shape[2]):
                        img[x:x + h, y:y + w, c] = random.uniform(0, 1)
                else:
                    img[x:x + h, y:y + w] = random.uniform(0, 1)
                img = Image.fromarray(img)

                return img


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
