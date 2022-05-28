import torch
from torch.utils.data import DataLoader
from torch import nn
from utils.tools import train, validate
from utils.dataset import create_datasets
from utils.net import EEGNet, resnet18
from torchvision import models
import warnings

_, _, testset = create_datasets(train='cls_train.txt', val='cls_val.txt', test='cls_test.txt')

batch_size = 128

test_loader = DataLoader(testset, batch_size, shuffle=True, drop_last=True)

device = torch.device('cuda:0')


net = EEGNet().to(device)

net.load_state_dict(torch.load('ckpt/eegnet/best.pth'))

acc_best_eegnet = validate(test_loader)

net.load_state_dict(torch.load('ckpt/eegnet/final.pth'))

acc_final_eegnet = validate(test_loader)


net = resnet18(num_classes=2).to(device)

net.load_state_dict(torch.load('ckpt/resnet18/best.pth'))

acc_best_eegnet = validate(test_loader)

net.load_state_dict(torch.load('ckpt/resnet18/final.pth'))

acc_final_eegnet = validate(test_loader)
