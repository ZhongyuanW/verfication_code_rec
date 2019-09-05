# -*- coding: utf-8 -*-
# @Time    : 8/24/19 6:23 AM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : config.py
# @Software: PyCharm

HOME = "/home/zhongyuan/datasets/VerifiCodeRef"
LR_STEP = (16000,30000)
CLASS = [
            ' ',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
            'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
            'w', 'x', 'y', 'z',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z',
         ]


CLASS_NUM = 62

BATCH_SIZE = 256
MAX_SIZE = 40000
LR = 1e-1
MOMENTUM = 0.9
