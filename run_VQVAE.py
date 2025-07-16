from __future__ import print_function
# import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from six.moves import xrange
import umap
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import argparse
import copy
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision.utils import make_grid
# from skimage.metrics import structural_similarity as ssim
import torch.optim.lr_scheduler as lr_scheduler

from PIL import Image
from classification_task import classification_Model
from construction_task import construction_Model
from modulation import QAM, PSK
from thop import profile
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_residual_hiddens = 32
num_residual_layers = 4
embedding_dim = 512
num_embeddings = 256
commitment_cost = 0.25
learning_rate = 1e-3
output_channel = 256
epochs = 200
batch_size = 512
psnr = 12.0


# Dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train = torchvision.datasets.CIFAR10(root="data", train=True, download=False, transform=transform_train)
test = torchvision.datasets.CIFAR10(root="data", train=False, download=False, transform=transform_test)
# print(len(train))
# print(len(test))
dataset_train_spl, _ = torch.utils.data.random_split(train, [50000, len(train) - 50000])
dataset_test_spl, _ = torch.utils.data.random_split(test, [10000, len(test) - 10000])
test_data_loader = torch.utils.data.DataLoader(dataset_test_spl, batch_size=1, shuffle=False, num_workers=2)  # 用于测试,batch_size原值为1000

# 功率约束
def PowerNormalize(z):
    z_square = torch.mul(z, z)
    power = torch.mean(z_square).sqrt()
    if power > 1:
        z = torch.div(z, power)
    return z

def train_trx(model, optimizer, criterion, images, target, mod):
    model.train()
    loss, y_hat, perplexity = model(images, mod=mod)
    loss_cross = criterion(y_hat, target) + loss
    # loss_cross = criterion(y_hat, target)
    pred = y_hat.argmax(dim=1, keepdim=True)
    accuracy = pred.eq(target.view_as(pred))

    optimizer.zero_grad()
    loss_cross.backward()
    optimizer.step()
    return accuracy

def test_trx(model, test_data_loader, mod):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, target in test_data_loader:
            images = images.to(device)
            target = target.to(device)
            loss, y_hat, perplexity = model(images, mod=mod)
            pred = y_hat.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        return correct / len(test_data_loader.dataset), perplexity

# 面向任务通信网络训练---分类任务
def main_train():
    print('面向任务通信网络训练')
    test_acc = 0
    kwargs = {'num_workers': 1, 'pin_memory': True}
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    mod = QAM(num_embeddings, psnr)
    model = classification_Model(num_embeddings, embedding_dim, commitment_cost, output_channel, psnr, mod=mod).to(device)
    # model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
    for epoch in range(epochs):
        print('This is {}-th epoch'.format(epoch))
        print('分类任务')
        train1_data_loader = torch.utils.data.DataLoader(dataset_train_spl, batch_size=batch_size, shuffle=True, **kwargs)
        total_correct = 0
        start_time = time.time()
        for n, (images, target) in enumerate(train1_data_loader):
            images = images.to(device)
            target = target.to(device)
            correct = train_trx(model, optimizer, criterion, images, target, mod=mod)
            total_correct += correct.sum().item()
        duration = time.time() - start_time
        print(duration)
        print("训练精度：", total_correct / len(train1_data_loader.dataset))
        acc, perplexity = test_trx(model, test_data_loader, mod)
        print('测试精度:', acc)
        scheduler.step()
        if acc > test_acc:
            test_acc = acc
            saved_model = copy.deepcopy(model.state_dict())
            with open('./Classification_results/CIFAR_SNR_{}_epoch_{}.pth'.format(psnr, epoch), 'wb') as f:
                torch.save({'model': saved_model}, f)

# 测试分类精度
def main_test(): # 测试分类精度
    mod = QAM(num_embeddings, psnr)
    model = classification_Model(num_embeddings, embedding_dim, commitment_cost, output_channel, psnr, mod=mod).to(device)
    model.load_state_dict(torch.load('256x512/CIFAR_SNR_12.0_epoch_199.pth')['model'])
    accuracy = 0
    t = 1
    start_time = time.time()
    for i in range(t):
        acc, perplexity = test_trx(model, test_data_loader, mod)
        accuracy += acc
        perplexity += perplexity
    print('测试精度:', accuracy / t)
    print('困惑度:', perplexity / t)
    duration = time.time() - start_time
    print(duration)

# 测试
def main_flops():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    mod = QAM(num_embeddings, psnr)
    model = classification_Model(num_embeddings, embedding_dim, commitment_cost, output_channel, psnr, mod=mod).to(device)
    model.load_state_dict(torch.load('256x512/CIFAR_SNR_12.0_epoch_199.pth')['model'])
    inputs = torch.randn(1, 3, 32, 32)
    inputs = inputs.to(device)
    inputs = inputs.cuda()
    flops, params = profile(model, inputs=(inputs,))
    print('flops:',flops / 1e9,'G')  # flops单位G，para单位M
    print('para:',params / 1e6,'M')  # flops单位G，para单位M

if __name__ == '__main__':
    # seed_torch(0)
    main_train() # 训练分类任务通信网络
    # main_test()  # 测试任务通信网络
    # main_flops()