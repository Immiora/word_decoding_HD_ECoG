'''
Multi-layered perception model
'''

import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=.25, inplace=True)
        self.droprate = drop_rate

    def forward(self, x):
        out = self.leaky_relu(self.bn(self.lin(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return out

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, n_blocks=3, drop_rate=0.0):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.hidden = hidden_channels
        self.droprate = drop_rate
        #self.blocks = []
        self.blocks = self._make_blocks(in_channels, hidden_channels, n_blocks, drop_rate)
        # for i in range(n_blocks):
        #     self.blocks.append(BasicBlock(in_channels, self.hidden, self.droprate))

        #self.block1 = BasicBlock(in_channels, self.hidden)
        #self.block2 = BasicBlock(self.hidden, self.hidden)
        #self.block3 = BasicBlock(self.hidden, self.hidden)
        self.lin = nn.Linear(self.hidden, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_blocks(self, in_channels, hidden_channels, n_blocks, drop_rate):
        blocks = []
        for i in range(n_blocks):
            if i == 0:
                blocks.append(BasicBlock(in_channels, hidden_channels, drop_rate))
            else:
                blocks.append(BasicBlock(hidden_channels, hidden_channels, drop_rate))
        return nn.Sequential(*blocks)

    def forward(self, x):
        """
            input: x (B x C x T)
        """
        out = x.reshape(-1, self.in_channels)
        for i in range(len(self.blocks)):
            out = self.blocks[i](out)
        #out = self.block1(out)
        #out = self.block2(out)
        #out = self.block3(out)
        out = self.relu(self.bn(self.lin(out))).unsqueeze(-1)
        return out


def make_model(in_dim, out_dim, **kwargs):
    model = MLP(in_dim, out_dim, **kwargs)
    return model

