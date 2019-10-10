import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import *

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate=12, depth=100, reduction=0.5,bottleneck=True, nClasses=(CLASS_NUM+1)*4):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=7, padding=1, stride=2,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        #self.conv_last = nn.Conv2d(nChannels,nClasses,1)
        #self.bn2 = nn.BatchNorm2d(nClasses)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #self.softmax=nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        # out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))

        out = F.relu(self.bn1(out),inplace=True)

        #out=F.relu(self.bn2(self.conv_last(out)),inplace=True)
        out=self.avgpool(out)
        out = torch.squeeze(out,dim=2).squeeze(dim=2)
        out = self.fc(out)

        #out = self.fc(out)
        # out_1, out_2, out_3, out_4 = out[:,:CLASS_NUM+1],out[:,CLASS_NUM+1:(CLASS_NUM+1)*2],out[:,(CLASS_NUM+1)*2:(CLASS_NUM+1)*3],out[:,(CLASS_NUM+1)*3:]
        #
        # out_1,out_2,out_3,out_4 = F.log_softmax(out_1,dim=1),F.log_softmax(out_2,dim=1),F.log_softmax(out_3,dim=1),F.log_softmax(out_4,dim=1)
        #
        # out = torch.cat((out_1,out_2,out_3,out_4),dim=1)

        return out


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    net = DenseNet()
    net = net.cuda()
    net.eval()

    x = torch.zeros((64,3,160,60),dtype = torch.float32)
    x=x.cuda()
    y = net(x)
    print(y.size())
