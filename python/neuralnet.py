import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class AlphaZeroNet(nn.Module):
    def __init__(self, out_channels):
        super(AlphaZeroNet, self).__init__()
        # shared layers
        self.conv = nn.Conv2d(2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)
        # policy layers
        self.pconv1 = nn.Conv2d(out_channels, 2, 1, bias=False)
        self.pbn = nn.BatchNorm2d(2)
        self.pfc = nn.Linear(2 * 6 * 7, 7)
        #  # value layers
        self.vconv1 = nn.Conv2d(out_channels, 1, 1, bias=False)
        self.vbn = nn.BatchNorm2d(1)
        self.vfc1 = nn.Linear(6 * 7, out_channels, bias=False)
        self.vfc2 = nn.Linear(out_channels, 1)

    # #2x1x1 conv -> bn -> relu -> fc
    def policy_head(self, x):
        out = self.pconv1(x) 
        out = self.pbn(out) 
        out = self.relu(out) 
        out = out.view(-1, 2 * 6 * 7)
        out = self.pfc(out) 
        return out


    # #1x1x1 conv -> bn -> relu -> fc -> relu -> fc -> tanh
    def value_head(self, x):
        out = self.vconv1(x) 
        out = self.vbn(out) 
        out = self.relu(out)
        out = out.view(-1, 1 * 6 * 7)
        out = self.vfc1(out) 
        out = self.relu(out)
        out = self.vfc2(out)
        return torch.tanh(out)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.res1(out)
        out = self.res2(out)
        return self.policy_head(out), self.value_head(out)
