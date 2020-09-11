import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import Bottleneck
import numpy as np


class ResNet(torchvision.models.resnet.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__(block, layers, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        for i in range(2, 5):
            getattr(self, 'layer%d'%i)[0].conv1.stride = (2,2)
            getattr(self, 'layer%d'%i)[0].conv2.stride = (1,1)


class Encoder(nn.Module):
    def __init__(self, resnet101_file):
        super(Encoder, self).__init__()
        resnet = ResNet(Bottleneck, [3, 4, 23, 3])
        ckpt = torch.load(resnet101_file, map_location=lambda s, l: s)
        resnet.load_state_dict(ckpt)
        self.resnet = resnet
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def preprocess(self, image):
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=2)

        image = image.astype('float32') / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))
        image = self.transforms(image)
        return image

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2).squeeze()
        att = F.adaptive_avg_pool2d(x, [att_size, att_size]).squeeze().permute(1, 2, 0)

        return fc, att
