# -*- coding: utf-8 -*-
# Author: Xingcheng Xu
from torchvision import datasets, transforms


root = "./data"

trans = transforms.Compose([transforms.ToTensor(),
               transforms.Lambda(lambda x: (x-0.5)*2) # Scale between [-1, 1]
               ])
train_set = datasets.FashionMNIST(root=root, train=True, transform=trans, download=True)
# train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
