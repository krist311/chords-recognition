import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 500)
        self.bn2 = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc4 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc5 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if (padding != 1) or (in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=3),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3,
                              stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.res_block1 = ResidualBlock(16, 16)
        self.res_block2 = ResidualBlock(16, 32, padding=0)
        self.res_block3 = ResidualBlock(32, 64, padding=0)
        self.avg_pool = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(14 * 14 * 64, 500)
        self.bn3 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)

        out = self.avg_pool(out)
        out = self.bn2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.bn3(out)
        out = self.fc2(out)
        return out
