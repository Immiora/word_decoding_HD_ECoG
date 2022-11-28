'''
https://github.com/andreasveit/densenet-pytorch
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, drop_rate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(inter_planes)
        self.conv2 = nn.Conv1d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, drop_rate=0.0, pool=2):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = drop_rate
        self.pool = pool

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool1d(out, self.pool)
        #return F.avg_pool1d(out, 3, stride=1, padding=1)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, drop_rate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, drop_rate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, in_channels, num_classes, depth=10, growth_rate=10,
                 reduction=1, bottleneck=False, drop_rate=0.0, pool=2):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate # 20
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)

        # 1st conv before any dense block
        self.conv1 = nn.Conv1d(in_channels, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, drop_rate)
        in_planes = int(in_planes+n*growth_rate)  # 40
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), drop_rate=drop_rate, pool=pool)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, drop_rate)
        in_planes = int(in_planes+n*growth_rate) # 60
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), drop_rate=drop_rate, pool=pool)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, drop_rate)
        in_planes = int(in_planes+n*growth_rate) # 80
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        #self.fc = nn.Linear(in_planes, num_classes)
        self.fc = nn.Conv1d(in_planes, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.shape[-1] < 12:
            p = 12 - x.shape[-1]
            p1 = int(p/2) # RuntimeError: Integer division of tensors using div or / is no longer supported, and in a future release div will perform true division as in Python 3. Use true_divide or floor_divide (// in Python) instead
            x = F.pad(x, ([p1, p-p1]))
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool1d(out, 3)
        #out = F.avg_pool1d(out, 3, stride=1, padding=1)
        #out = out.view(-1, self.in_planes)
        return self.fc(out)


def make_model(in_dim, out_dim, **kwargs):
    model = DenseNet3(in_dim, out_dim, **kwargs)
    return model

