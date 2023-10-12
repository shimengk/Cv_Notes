import torch
import torch.nn as nn
import torchsummary


class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            # 12 * 12 * 3
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(10),
            nn.PReLU(),

            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        # 3 个输出层：置信度、偏移量、关键点
        self.conf_out = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.offset_out = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)
        self.landmark_out = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1)

    def forward(self, x):
        out = self.layer(x)
        # 置信度的输出层：概率，输出函数是sigmoid（）
        # BCELoss()：必须经过sigmoid的激活()
        conf = torch.sigmoid(self.conf_out(out))
        offset = self.offset_out(out)
        landmark = self.landmark_out(out)
        return conf, offset, landmark


class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(28),
            nn.PReLU(),

            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(48),
            nn.PReLU(),

            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.conf_out_layer = nn.Sequential(
            # 展平
            nn.Linear(in_features=3 * 3 * 64, out_features=128),
            nn.PReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        self.offset_out_layer = nn.Sequential(
            nn.Linear(in_features=3 * 3 * 64, out_features=128),
            nn.PReLU(),
            nn.Linear(in_features=128, out_features=4)
        )
        self.landmark_out_layer = nn.Sequential(
            nn.Linear(in_features=3 * 3 * 64, out_features=128),
            nn.PReLU(),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        # 卷积主干网络：out：NCHW
        out = self.layer(x)
        # return out
        # NCHW --> NV
        out = out.reshape(-1, 3 * 3 * 64)
        # conf  sigmoid()
        conf = torch.sigmoid(self.conf_out_layer(out))
        offset = self.offset_out_layer(out)
        landmark = self.landmark_out_layer(out)
        return conf, offset, landmark


class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=128 * 3 * 3, out_features=256),
            nn.PReLU()
        )

        # 3输出层
        self.conf_out_layer = nn.Linear(in_features=256, out_features=1)
        self.offset_out_layer = nn.Linear(in_features=256, out_features=4)
        self.landmark_out_layer = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # 主干网络：卷积
        out = self.conv_layer(x)
        # 主干网络：全连接
        # 展平
        out = out.reshape(-1, 128 * 3 * 3)
        out = self.linear_layer(out)
        # 输出参数
        conf = torch.sigmoid(self.conf_out_layer(out))
        offset = self.offset_out_layer(out)
        landmark = self.landmark_out_layer(out)
        return conf, offset, landmark


if __name__ == '__main__':
    print('PNet')
    torchsummary.summary(PNet().cuda(), input_size=(3, 12, 12))
    # torchsummary.summary(RNet().cuda(), input_size=(3, 24, 24))
    # torchsummary.summary(ONet().cuda(), input_size=(3, 48, 48))
