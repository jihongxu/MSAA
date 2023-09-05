import torch

from V2.model_utils.models import *
from torchsummary import summary

model_name = 'MobileNetV3_att_NAM'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(model_name, num_class=5, pretrained=True, device=device)
stat(model, input_size=[3, 256, 256])  # 查看网络参数
print(model)

# device = torch.device("cpu")
model = model.to(device)
summary(model, input_size=(3, 256, 256))

