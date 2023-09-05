from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from V2.model_utils.model.MobileNetv3_x5_x7_mish import *
from V2.model_utils.model.MobileNetv3_x5x7_ECA_v4 import *
from V2.model_utils.model.MobileNetv3_x5x7_ECA_v3 import *
from V2.model_utils.model.MobileNetv3_x5x7_ECA_v2 import *
from V2.model_utils.model.MobileNetv3_x5x7_ECA import *
from V2.model_utils.model.MobileNetv3_test import *
from V2.model_utils.model.MobileNetv3_mish_test import *

from V2.model_utils.model.MobileNetv3_att_eca import *
from V2.model_utils.model.MobileNetv3_att_cbam import *
from V2.model_utils.model.MobileNetv3_att_cbam2 import *
from V2.model_utils.model.MobileNetv3_att_ca import *
from V2.model_utils.model.MobileNetv3_att_dan import *
from V2.model_utils.model.MobileNetv3_att_se import *
from V2.model_utils.model.MobileNetv3_att_psa import *
from V2.model_utils.model.MobileNetv3_att_sge import *
from V2.model_utils.model.MobileNetv3_att_sk import *
from V2.model_utils.model.MobileNetv3_att_ccnet import *
from V2.model_utils.model.MobileNetv3_att_A2 import *
from V2.model_utils.model.MobileNetv3_att_eca_mish import *
from V2.model_utils.model.MobileNetv3_att_MobileViT import *
from V2.model_utils.model.MobileNetv3_att_Vip import *
from V2.model_utils.model.MobileNetv3_x7x10 import *
from V2.model_utils.model.MobileNetv3_x7x10_mish import *
from V2.model_utils.model.MobileNetv3_x7x10_noAtt import *
from V2.model_utils.model.MobileNetv3_x7x10_noAtt_mish import *
from V2.model_utils.model.MobileNetv3_att_gam import *

"""
    MobilNet v3、ShuffNet 、MobilNet v2、AlexNet v1、AlexNet v2
    如果你需要用预训练模型，设置pretrained=True
    如果你不需要用预训练模型，设置pretrained=False，默认是False，你可以不写
"""


class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


def get_model(model_name, num_class=5, pretrained=True, device=None):

    if model_name == 'MobileNetV3_mish_test':  # 测试激活函数mish替换relu
        model = MobileNetV3_mish_test(num_class=num_class, pretrained=pretrained)

    elif model_name == 'MobileNetV3_att_eca':  # 测试eca注意力机制代替se
        model = MobileNetV3_att_eca(num_class=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetV3_att_cbam':  # 测试cbam注意力机制代替se
        model = MobileNetV3_att_cbam(num_class=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetV3_att_cbam2':  # 测试cbam注意力机制代替se
        model = MobileNetV3_att_cbam2(num_class=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetV3_att_ca':  # 测试ca注意力机制代替se
        model = MobileNetV3_att_ca(num_class=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetV3_att_dan':  # 测试dan注意力机制代替se
        model = MobileNetV3_att_dan(num_class=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetV3_att_se':  # 测试dan注意力机制代替se
        model = MobileNetV3_att_se(num_class=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetV3_att_psa':  # 测试dan注意力机制代替se
        model = MobileNetV3_att_psa(num_class=num_class, pretrained=pretrained, device=device)
    elif model_name == 'MobileNetV3_att_sge':  # 测试dan注意力机制代替se
        model = MobileNetV3_att_sge(num_class=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetV3_att_sk':  # 测试dan注意力机制代替se
        model = MobileNetV3_att_sk(num_class=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetV3_att_ccnet':  # 测试dan注意力机制代替se
        model = MobileNetV3_att_ccnet(num_class=num_class, pretrained=pretrained, device=device)
    elif model_name == 'MobileNetV3_att_A2':  # 不太行
        model = MobileNetV3_att_A2(num_class=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetV3_att_eca_mish':  #
        model = MobileNetV3_att_eca_mish(num_class=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetV3_att_MobileVit':  #
        model = MobileNetV3_att_MobileVit(num_class=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetV3_att_Vip':  #
        model = MobileNetV3_att_Vip(num_class=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetv3_att_gam':  #
        model = MobileNetV3_att_gam(num_class=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetV3_att_NAM':  #
        from V2.model_utils.model.MobileNetv3_att_NAM import MobileNetV3_att_NAM
        model = MobileNetV3_att_NAM(num_class=num_class, pretrained=pretrained)

    elif model_name == 'MobileNetV3_x7_x10':
        model = MobileNetV3_x7_x10(num_classes=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetV3_x7_x10_mish':
        model = MobileNetV3_x7_x10_mish(num_classes=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetV3_x7_x10_noAtt':
        model = MobileNetV3_x7_x10_noAtt(num_classes=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetV3_x7_x10_noAtt_mish':
        model = MobileNetV3_x7_x10_noAtt_mish(num_classes=num_class, pretrained=pretrained)

    elif model_name == 'MobileNetv3':
        model = models.mobilenet_v3_large(pretrained=pretrained)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_class)
    elif model_name == 'MobileNetv3_test':
        model = MobileNetV3_test(num_classes=num_class, pretrained=pretrained)
    elif model_name == 'ShuffNet':
        model = models.shufflenet_v2_x1_5(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_class)
    elif model_name == 'MobileNetv2':
        model = models.mobilenet_v2(pretrained=pretrained)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_class)
    elif model_name == 'AlexNetV1':
        model = models.alexnet(pretrained=pretrained)
        model.classifier[-1] = nn.Linear(4096, num_class)
    elif model_name == 'AlexNetV2':
        model = torchvision.models.AlexNet(num_classes=num_class)
        if pretrained:
            pretrained_dict = torchvision.models.alexnet(pretrained=True).state_dict()
            model_dict = model.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            model.load_state_dict(model_dict)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        # 用于处理分类任务的最后一层
        model.fc = nn.Linear(in_features=2048, out_features=num_class, bias=False)

    elif model_name == 'MobileNetv3_ECA_x5x7_v2':
        model = MobileNetV3_Large_Fusion_v2(num_classes=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetv3_ECA_x5x7':
        model = MobileNetV3_Large_Fusion(num_classes=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetv3_ECA_x5x7_v3':
        model = MobileNetV3_Large_Fusion_v3(num_classes=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetv3_ECA_x5x7_v4':
        model = MobileNetV3_Large_Fusion_v4(num_classes=num_class, pretrained=pretrained)
    elif model_name == 'MobileNetv3_x5_x7_mish':
        model = MobileNetV3_Large_Fusion_mish(num_classes=num_class, pretrained=pretrained)
    else:
        raise ValueError("Invalid model name. Choose among MobileNetv3, ShuffNet, MobileNetv2, AlexNetV1_pad, AlexNetV2")

    return model
