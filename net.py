import torch
from torch import nn


class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        self.pose_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.pose_net(x)
        return x


class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.face_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(1568, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.face_net(x)
        return x


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.pose_net = PoseNet()
        self.head_net = nn.Flatten()
        self.face_net = FaceNet()
        self.fusion_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2659, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )

    def forward(self, x, y, z):
        x = self.pose_net(x)
        y = self.face_net(y)
        z = self.head_net(z)
        out = torch.cat((x, y), 1)
        out = torch.cat((out, z), 1)
        out = self.fusion_net(out)
        return out
