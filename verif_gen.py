# -*- coding: utf-8 -*-
# @Time    : 5/23/19 11:18 AM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : verif_gen.py
# @Software: PyCharm

import os
from captcha.image import ImageCaptcha
from random import randint
from config import *

def verfiCodeGen(length=4, images=50000, type="train"):
    labels = ""
    for m in range(0,images):
        chars = ''
        for i in range(length):
            chars += CLASS[randint(0, CLASS_NUM)]
        image = ImageCaptcha().generate_image(chars)
        image.save(os.path.join(HOME, "images/%s/%i.jpg"%(type,m)))
        labels += chars+"\n"
        print("%i/%i completed!"%(m,images))
    with open(os.path.join(HOME,"labels/%s/label.txt" % (type)), "w") as f:
        f.write(labels)
    return 0

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    verfiCodeGen(images=50000,type="train")