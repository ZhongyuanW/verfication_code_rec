# -*- coding: utf-8 -*-
# @Time    : 8/24/19 11:49 AM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : train.py
# @Software: PyCharm

import net as Net
import dataset as Dataset
import torch.utils.data.dataloader as Dataloader
import torch.nn as nn
import torch.optim as optim
import time
import torch
import visdom
from torch.autograd import Variable
from config import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

viz = visdom.Visdom()

def train():

    dataset = Dataset.verDataset(phase="train")

    print("dataset size is: %d"%dataset.__len__())

    dataloader = Dataloader.DataLoader(dataset, batch_size=BATCH_SIZE,
                                       shuffle=True, num_workers=4,drop_last=True)

    batch_iterator = iter(dataloader)

    net = Net.DenseNet()
    net.train()

    net = nn.DataParallel(net)
    net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    net.train()

    vis_title = dataset.name
    vis_legend = ["loss"]
    iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)

    step_index = 0

    for i in range(MAX_SIZE):
        t0 = time.time()

        if i in LR_STEP:
            step_index += 1
            adjust_learning_rate(optimizer,step_index)


        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(dataloader)
            images, targets = next(batch_iterator)

        images,targets = Variable(images),Variable(targets)

        optimizer.zero_grad()

        images,targets = images.cuda(),targets.cuda()
        y = net(images)

        y1,y2,y3,y4 = y[:,:CLASS_NUM+1],y[:,CLASS_NUM+1:(CLASS_NUM+1)*2],y[:,(CLASS_NUM+1)*2:(CLASS_NUM+1)*3],y[:,(CLASS_NUM+1)*3:]
        target_1,target_2,target_3,target_4 = targets[:,0].squeeze(),targets[:,1].squeeze(),targets[:,2].squeeze(),targets[:,3].squeeze()

        # print(target_1)

        loss_1 = criterion(y1, target_1)
        loss_2 = criterion(y2, target_2)
        loss_3 = criterion(y3, target_3)
        loss_4 = criterion(y4, target_4)
        loss = loss_1+loss_2+loss_3+loss_4
        # print("loss_1 ",loss_1,"loss_2 ",loss_2,"loss_3 ",loss_3,"loss_4 ",loss_4)

        # loss = criterion(y,targets)

        loss.backward()
        optimizer.step()

        precision_1 = torch.eq(torch.argmax(y1,dim=1).squeeze(),target_1)
        precision_2 = torch.eq(torch.argmax(y2, dim=1).squeeze(), target_2)
        precision_3 = torch.eq(torch.argmax(y3, dim=1).squeeze(), target_3)
        precision_4 = torch.eq(torch.argmax(y4, dim=1).squeeze(), target_4)

        # print(precision_1,precision_2,precision_3,precision_4)

        # print(target_1,torch.argmax(y1,dim=1).squeeze())

        precision = torch.eq(torch.eq(torch.eq(precision_1,precision_2),precision_3),precision_4).float().sum()/BATCH_SIZE

        t1 = time.time()

        if i %10 == 0:
            print("[%d/%d] timer %.4f, loss %.4f, prec %.2f%%"%(i,MAX_SIZE,t1-t0,loss.item(),precision.item()*100))

        update_vis_plot(i, loss.item(),iter_plot, 'append')

        if i != 0 and i % 1000 == 0:
            print("save state, iter: %d"%i)

            torch.save(net.state_dict(),"weights/densenet121_%d.pth"%(i))

    torch.save(net.state_dict(), "weights/densenet121_final.pth")


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loss, window, update_type,epoch_size=1):
    viz.line(
        X=torch.ones((1,)).cpu() * iteration,
        Y=torch.Tensor([loss]).unsqueeze(0).cpu() / epoch_size,
        win=window,
        update=update_type
    )

def adjust_learning_rate(optimizer, step, gamma=0.1):
    lr = LR * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    train()



