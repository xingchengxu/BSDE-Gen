# -*- coding: utf-8 -*-
# Author: Xingcheng Xu
"""
# FBSGE-Gen Model
"""

"""
## Model Class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


# Setting reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
# setup_seed(0)


class FBSDEGen(nn.Module):
    def __init__(self, b, sigma, f, dim_x, dim_w, dim_y=784, dim_h1=1000, dim_h2=600, dim_h3=1000, T=1.0, N=200, device=None):
        super(FBSDEGen, self).__init__()

        self.b = b
        self.sigma = sigma
        self.f = f
        self.T = T
        self.N = N
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_w = dim_w

        self.y0_nn = nn.Sequential(
            nn.Linear(dim_x, dim_h1),
            # nn.Dropout(p=0.2),
            nn.GELU(),
            nn.Linear(dim_h1, dim_h2),
            nn.Dropout(p=0.2),
            nn.GELU(),
            nn.Linear(dim_h2, dim_h3),
            nn.Dropout(p=0.2),
            nn.GELU(),
            nn.Linear(dim_h3, dim_y)
        )

        self.z_nn = nn.Sequential(
            nn.Linear(dim_x + 1, dim_h1),
            # nn.Dropout(p=0.2),
            nn.GELU(),
            nn.Linear(dim_h1, dim_h2),
            nn.Dropout(p=0.2),
            nn.GELU(),
            nn.Linear(dim_h2, dim_h3 * dim_w // 4),
            nn.Dropout(p=0.2),
            nn.GELU(),
            nn.Linear(dim_h3 * dim_w // 4, dim_y * dim_w)
        )

    def forward(self, input_samples):
        batch_size = input_samples.size()[0]

        delta_t = self.T / self.N

        y_0 = self.y0_nn(input_samples).to(device)

        x = input_samples + torch.zeros(batch_size, self.dim_x, device=device)
        y = y_0 + torch.zeros(batch_size, self.dim_y, device=device)

        for i in range(self.N):
            z_input = torch.cat((x, torch.ones(batch_size, 1, device=device) * delta_t * i), 1)
            z = self.z_nn(z_input).reshape(-1, self.dim_y, self.dim_w).to(device)

            dw = torch.randn(batch_size, self.dim_w, 1, device=device) * np.sqrt(delta_t)
            x = x + self.b(delta_t * i, x) * delta_t + torch.matmul(self.sigma(delta_t * i, x), dw).reshape(-1, self.dim_x)
            y = y - self.f(delta_t * i, x, y, z) * delta_t + torch.matmul(z, dw).reshape(-1, self.dim_y)
        return x, y


# *************************************************** #
"""
## Model Training
### DDP: DistributedDataParallel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os


def train(model, train_loader, optimizer, n_epochs, device, world_size, dim_in, store_path):
    model.train()
    # criterion = torch.nn.MSELoss().to(device)  # MSELoss, SmoothL1Loss; KLDivLoss, MMD
    best_loss = float("inf")

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for batch_idx, (img, _) in enumerate(train_loader):
            optimizer.zero_grad()

            n_img = img.size()[0]
            img = img.view(n_img, -1)
            img = Variable(img).to(device)

            input_noise = torch.randn((n_img, dim_in), device=device)
            _, y = model(input_noise)
            # loss = criterion(y, img)
            loss = MMD(y, img)
            
            loss.backward()
            optimizer.step()

            # if dist.get_rank() == 0:
            #     print(f"epoch={epoch + 1}, batch_idx={batch_idx + 1}, loss={loss:.4f}")

            epoch_loss += loss.item()

        epoch_loss /= (batch_idx + 1)

        # use torch.distributed.all_reduce to average the loss across all GPUs
        loss_tensor = torch.tensor(epoch_loss).to(device)
        torch.distributed.all_reduce(loss_tensor)
        epoch_loss = loss_tensor.item() / world_size

        log_string = f"Loss at epoch {epoch+1}: {epoch_loss:.4f}"

        # Storing the model
        if dist.get_rank() == 0 and best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(model.module.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        if dist.get_rank() == 0:
            print(log_string)


def MMD(x, y, kernel="multiscale"):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


# 打印gpu数量
print("Using", torch.cuda.device_count(), "GPUs")

# initialize process group
dist.init_process_group(backend="nccl", init_method='env://')

# set rank and size
rank = int(os.environ['LOCAL_RANK'])
world_size = dist.get_world_size()
print(f"rank={rank}, world_size={world_size}")

# 根据rank指定使用哪块gpu
torch.cuda.set_device(rank)

# 定义设备，根据gpu的数量来设定，初始gpu为0，这里gpu数量为8
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
device = torch.device("cuda", rank)

# 把模型加载到cuda上
# hyperparameters
T, N = 1.0, 200
dim_x, dim_y, dim_w = 32, 784, 32
dim_h1, dim_h2, dim_h3 = 64, 32, 64 # 1000, 600, 1000

n_epochs = 10000
batch_size = 512
lr = 0.0001

root = "./data"
store_path="fbsde_mlp_ddp_0322.pt"
last_store_path="fbsde_mlp_ddp_0322_last.pt"

# X: Ornstein–Uhlenbeck (OU) process
def b(t, x):
    batch_size = x.size()[0]
    return -x.reshape(batch_size, dim_x).to(device)


def sigma(t, x):
    batch_size = x.size()[0]
    idmat = torch.eye(dim_x, dim_w, device=device)*np.sqrt(2)
    return idmat.repeat(batch_size, 1, 1)


# # X: Brownian Motion
# def b(t, x):
#     batch_size = x.size()[0]
#     return torch.zeros(batch_size, dim_x, device=device)
#
#
# def sigma(t, x):
#     batch_size = x.size()[0]
#     idmat = torch.eye(dim_x, dim_w, device=device)
#     return idmat.repeat(batch_size, 1, 1)


# def f(t, x, y, z):
#     # generator, it can be choosen!
#     batch_size = x.size()[0]
#     return -torch.abs(y).reshape(batch_size, dim_y).to(device)


def f(t, x, y, z):
    # generator, it can be choosen!
    batch_size = x.size()[0]
    z_term = torch.sum(torch.abs(z), 2).reshape(batch_size, dim_y).to(device)
    return (-y+z_term).reshape(batch_size, dim_y).to(device)


# def f(t, x, y, z):
#     # generator, it can be choosen! Interaction of components, linear approximation.
#     import torch
#     A = torch.load('tensor_A.pt').to(device)
#     batch_size = x.size()[0]
#     B = (-y).reshape(batch_size, dim_y, 1).to(device)
#     return torch.matmul(0.01*A, B).reshape(batch_size, dim_y).to(device)


model = FBSDEGen(b, sigma, f, dim_x, dim_w, dim_y, dim_h1, dim_h2, dim_h3, T, N, device=device)
model = model.to(device)

if dist.get_rank() == 0:
    n_params = sum([p.numel() for p in model.parameters()])
    print(f"number of parameters: {n_params}")

try:
    model.load_state_dict(torch.load(store_path))
    print("#### Model parameters are loaded. ####")
except:
    pass

# 初始化DDP模型
model = DDP(model, device_ids=[rank], output_device=rank)

# 数据分到各gpu上, dataloader
trans = transforms.Compose([transforms.ToTensor(),
               transforms.Lambda(lambda x: (x-0.5)*2) # Scale between [-1, 1]
               ])
train_set = datasets.FashionMNIST(root=root, train=True, transform=trans, download=True)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    num_workers=world_size,
    pin_memory=True,
    sampler=train_sampler,
    shuffle=False)

optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train(model, train_loader, optimizer, n_epochs, device, world_size, dim_x, store_path)

if dist.get_rank() == 0:
    torch.save(model.module.state_dict(), last_store_path)
    print("Training Completed!")

dist.destroy_process_group()

"""**END**"""
