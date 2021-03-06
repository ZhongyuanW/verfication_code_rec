# -*- coding: utf-8 -*-
# @Time    : 9/3/19 8:48 AM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : eval.py
# @Software: PyCharm

import net as Net
import dataset as Dataset
from torch.autograd import Variable
from config import *
import torch
import torch.nn as nn

def eval(nums,weight_path=None,net=None,phase="test"):
    correct = 0
    dataset = Dataset.verDataset(phase="test")
    if phase=="test":
        net = Net.DenseNet()
        net = nn.DataParallel(net)
        net = net.cuda()
        weighted = torch.load(weight_path)
        print(weighted)
        net.load_state_dict(weighted)
        print("loading weight!")
    net.eval()

    for i in range(nums):
        image, label = dataset.__getitem__(i)
        image = Variable(image.unsqueeze(0)).cuda()
        precision = net(image).squeeze(0).cpu().data

        precision = decode(precision)

        if label == precision:
            correct += 1
        if phase=="test":
            print("%d completed."%(i+1))

    print("correct: %d, precision: %.2f%%" % (correct, correct / nums * 100))
    if phase == "train":
        net.train()
    return (correct / nums)

def decode(precision):
    label1,label2,label3,label4 = precision[:CLASS_NUM+1],\
            precision[CLASS_NUM+1:(CLASS_NUM+1)*2],precision[(CLASS_NUM+1)*2:(CLASS_NUM+1)*3],precision[(CLASS_NUM+1)*3:]
    label1,label2,label3,label4 = label1.numpy().argmax(),label2.numpy().argmax(),label3.numpy().argmax(),label4.numpy().argmax()
    return CLASS[label1]+CLASS[label2]+CLASS[label3]+CLASS[label4]


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    nums = 1000
    weight_path = r"weights/densenet121_1000.pth"
    correct = eval(nums, weight_path)
