import torch.nn as nn
import math
import torch


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes,
                               kernel_size=11,
                               bias=False,
                               padding=5)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = nn.Conv2d(planes, planes,
                               kernel_size=15,
                               stride=stride,
                               padding=7,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=11,
                               bias=False,
                               padding=5)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        residual = out
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class ECGNet(nn.Module):
    def __init__(self, block, layers, num_classes=34):
        self.inplanes = 16
        super(ECGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16,
                               kernel_size=19,
                               stride=2,
                               padding=9,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=2, padding=3)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.globavgpool = nn.AdaptiveAvgPool2d(1)
        self.globmaxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)
        self.fc1 = nn.Linear(128 * block.expansion, 128 * block.expansion)
        self.fc_atten_1 = nn.Linear(128 * block.expansion,
                                    128 * block.expansion)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def attention(self, x, i):
        out = torch.transpose(x, 1, 3)
        out = self.fc_atten_1(out)
        out = torch.transpose(out, 1, 3)
        weight = self.globavgpool(out)
        out = weight * x
        return out

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention(x, 0)
        x = torch.transpose(x, 1, 3)
        x = self.fc1(x)
        x = torch.transpose(x, 1, 3)
        y = self.globmaxpool(x)
        y = y.view(y.size(0), -1)
        x = self.globavgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, y], dim=1)
        x = self.fc(x)
        return x


def myecgnet(pretrained=False, **kwargs):
    model = ECGNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model
