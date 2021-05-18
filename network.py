import torch.nn as nn

from layer import Conv2d, AvgPool2d, Linear, IF2d, IF1d


class TailoredCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        in_channels = 3
        out_channels = 64
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(out_channels, out_channels, 5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(out_channels, out_channels, 3, bias=False),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            nn.ReLU(),
            nn.Linear(out_channels, num_classes, bias=False)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class SpikingCNN:
    def __init__(self):
        self.conv1 = Conv2d(3, 64, 5)
        self.if1 = IF2d(64, 20, 20, threshold=5)
        self.sub1 = AvgPool2d(2, 2)
        self.conv2 = Conv2d(64, 64, 5)
        self.if2 = IF2d(64, 6, 6, threshold=0.99)
        self.sub2 = AvgPool2d(2, 2)
        self.conv3 = Conv2d(64, 64, 3)
        self.if3 = IF2d(64, 1, 1)
        self.fc1 = Linear(64, 64)
        self.if4 = IF1d(64, threshold=0.99)
        self.fc2 = Linear(64, 10)
        self.if5 = IF1d(10)

    def __call__(self, x):
        o = self.if1(self.conv1(x))
        o = self.sub1(o)
        o = self.if2(self.conv2(o))
        o = self.sub2(o)
        o = self.if3(self.conv3(o))
        o = o.flatten()
        o = self.if4(self.fc1(o))
        o = self.if5(self.fc2(o))
        return o

