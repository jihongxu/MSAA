from fightingcv_attention.attention.CoordAttention import CoordAtt
import torch
from torch import nn
import shortuuid
"""对图片数据进行增强"""


def random_string():
    letters = shortuuid.ShortUUID(alphabet='0123456789abcdef').random(length=10)
    return letters


print(random_string())