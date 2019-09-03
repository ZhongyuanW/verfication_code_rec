# -*- coding: utf-8 -*-
# @Time    : 9/3/19 7:59 AM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : test.py
# @Software: PyCharm

import net as Net
from torch.autograd import Variable
import numpy as np
from config import *
import cv2
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test(path = r"/home/zhongyuan/datasets/VerifiCodeRef/images/test/1258.jpg"):
    net = Net.DenseNet()
    net = nn.DataParallel(net)
    net = net.cuda()
    weighted = torch.load(r"/home/zhongyuan/Workspace/verfication_code_rec/weights/densenet121_10000_7180.pth")
    # print(weighted)
    net.load_state_dict(weighted)
    print("load weight completed!")
    net.eval()

    image = cv2.imread(path)
    image = cv2.resize(image, (32, 32))
    max = image.max()
    min = image.min()
    image = (image - min) / (max - min)

    image = np.transpose(image,(2,0,1))

    image = torch.from_numpy(image).float()
    transform = transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    image = transform(image)

    image = Variable(image.unsqueeze(0)).cuda()

    precision = net(image).cpu().data

    code = decode(precision)

    print(code)


def decode(precision):
    label1,label2,label3,label4 = precision[:CLASS_NUM+1],\
            precision[CLASS_NUM+1:(CLASS_NUM+1)*2],precision[(CLASS_NUM+1)*2:(CLASS_NUM+1)*3],precision[(CLASS_NUM+1)*3:]
    label1,label2,label3,label4 = label1.numpy().argmax(),label2.numpy().argmax(),label3.numpy().argmax(),label4.numpy().argmax()
    return CLASS[label1]+CLASS[label2]+CLASS[label3]+CLASS[label4]


if __name__ == "__main__":
    test()