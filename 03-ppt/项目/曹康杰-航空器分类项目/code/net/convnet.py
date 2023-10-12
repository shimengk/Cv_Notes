import torch
import torch.nn as nn


# 定义基本的ResNet块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        stride = 1 if in_channels == out_channels else 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        # 如果输入输出通道不相等，表示需要 经shortcut结构使输入x与输出形状相匹配
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        return self.relu(out)


# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, classes_num=1000):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 添加残差块
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64))
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128),
            BasicBlock(128, 128))
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256),
            BasicBlock(256, 256))
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512),
            BasicBlock(512, 512))

        # 自适应平均池化
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，使out尺寸为classes_num
        self.fc = nn.Linear(512, classes_num)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxPool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgPool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


# 7 * 7 的卷积， 3 个 3 * 3 的卷积替代
class ResNet2(nn.Module):
    def __init__(self, classes_num=1000):
        super(ResNet2, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 添加残差块
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64))
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128),
            BasicBlock(128, 128))
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256),
            BasicBlock(256, 256))
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512),
            BasicBlock(512, 512))

        # 自适应平均池化
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，使out尺寸为classes_num
        self.fc = nn.Linear(512, classes_num)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.maxPool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgPool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


class ResNet3(nn.Module):
    def __init__(self, classes_num=1000):
        super(ResNet3, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 添加残差块
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64))
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128),
            BasicBlock(128, 128))
        # self.layer3 = nn.Sequential(
        #     BasicBlock(128, 256),
        #     BasicBlock(256, 256))
        # self.layer4 = nn.Sequential(
        #     BasicBlock(256, 512),
        #     BasicBlock(512, 512))

        # 自适应平均池化
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，使out尺寸为classes_num
        self.fc = nn.Linear(128, classes_num)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxPool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)

        out = self.avgPool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

class ResNet4(nn.Module):
    def __init__(self, classes_num=1000):
        super(ResNet4, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 添加残差块
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64))
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128),
            BasicBlock(128, 128))
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256),
            BasicBlock(256, 256))
        # self.layer4 = nn.Sequential(
        #     BasicBlock(256, 512),
        #     BasicBlock(512, 512))

        # 自适应平均池化
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，使out尺寸为classes_num
        self.fc = nn.Linear(256, classes_num)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.maxPool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)

        out = self.avgPool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
def resnet18(classes_num, version):
    """
    :param classes_num: 分类数目
    :param version: 版本
        v1：原始resnet18
        v2：`7*7`卷积换为3个`3*3`的卷积核
        v3: 去掉layer4的残差块
    :return:
    """
    return eval(f'ResNet{version}({classes_num})')

