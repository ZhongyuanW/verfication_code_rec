# -*- coding: utf-8 -*-
# @Time    : 5/29/19 11:42 AM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : dataset.py
# @Software: PyCharm

import os
import torch.utils.data.dataset as Dataset
import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
from config import *

class verDataset(Dataset.Dataset):
    def __init__(self,data_path=HOME,phase="train",transforms = transforms, name = "verfication_code_rec"):
        self.name = name
        self.phase = phase
        image_path = os.path.join(data_path,"images")
        label_path = os.path.join(data_path, "labels")
        with open(os.path.join(label_path,phase,"label.txt")) as f:
            labels = f.readlines()

        images_path_list = os.listdir(os.path.join(image_path,phase))

        images_target = []
        labels_target = []

        for i in images_path_list:
            path = os.path.join(image_path, phase,i)
            image = cv2.imread(path)

            #image = cv2.resize(image,(32,32))

            #max = image.max()
            #min = image.min()

            #image = (image-min)/(max-min)
            # print(image)

            images_target.append(image)

            index = int(i.split(".")[0])

            labels_target.append(labels[index])

        self.data = images_target
        self.label = labels_target

        self.transforms = transforms.Compose([
            #transforms.ToPILImage(),
            #transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(10.0, 10.0, 2.0, 0.2),
            #transforms.Resize((32,32)),
            #transforms.ToTensor(),
            #transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ])

        # print("loading data!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]

        data = np.transpose(data,(2,0,1))

        data = torch.from_numpy(data).float()
        #data = pre.preprocess(data)

        #data = self.transforms(data)

        label = self.label[item].replace("\n","").strip()

        if self.phase == "test":
            return data,label

        target = torch.zeros((4),dtype = torch.long)

        for i,code in enumerate(label):
            index = CLASS.index(code)
            target[i] = float(index)

        return data, target



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    import torch.utils.data.dataloader as Dataloader

    dataset = verDataset()

    dataloader = Dataloader.DataLoader(dataset,batch_size=4,
                    shuffle=True,num_workers=4,drop_last=True)

    batch_iterator = iter(dataloader)

    for i in range(100000):
        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(dataloader)
            images, targets = next(batch_iterator)
        print(targets[0])

