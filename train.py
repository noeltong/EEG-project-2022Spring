import torch
from torch.utils.data import DataLoader
from torch import nn
from utils.tools import train, validate
from utils.dataset import create_datasets
from utils.net import EEGNet, resnet18, resnet_1, googlenetlike
from torchvision import models
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda:0')

# ----------------------------------------
#    Loading Data
# ----------------------------------------

print("Loading data...")

trainset, valset, testset = create_datasets(train='cls_train.txt', val='cls_val.txt', test='cls_test.txt')

batch_size = 64

train_loader = DataLoader(trainset, batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(testset, batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(valset, batch_size, shuffle=True, drop_last=True)

print(f'Complete!\nNum of train set: {len(trainset)}\nNum of val set: {len(valset)}\nNum of testset: {len(testset)}')

# ----------------------------------------
#    Define Nets
# ----------------------------------------

# EEGNet

net = EEGNet().to(device)
task_name = 'EEGNet'

# ResNet 18

# net = resnet18(num_classes=2).to(device)
# task_name = 'resnet18'

# 4-block resnet

# net = resnet_1(num_classes=2).to(device)
# task_name = 'shallowResNet'

# inception models

# net = googlenetlike(out_channels=[16, 32, 64, 128], num_classes=2).to(device)
# task_name = 'our_proposed'

print('\n', task_name, '\n')

print(net)

# ----------------------------------------
#   eval 0
# ----------------------------------------

acc = validate(val_loader, net, device)
print(f'Initial acc: {100*acc:4f}%')

# ----------------------------------------
#    Define Loss and updater
# ----------------------------------------

loss = nn.CrossEntropyLoss()
updater = torch.optim.AdamW(net.parameters(), lr=0.005)

# ----------------------------------------
#    Training
# ----------------------------------------

torch.cuda.empty_cache()

num_epochs = 150
scheduler = torch.optim.lr_scheduler.ExponentialLR(updater, gamma=0.95)
train(net=net, train_iter=train_loader, test_iter=val_loader, loss=loss, updater=updater, num_epochs=num_epochs, device=device, scheduler=scheduler, save=True, task_name=task_name)

# ----------------------------------------
#   Calculate acc on test dataset
# ----------------------------------------

acc_test = validate(test_loader, net, device)

print(f'Final acc on test set: {100*acc_test:4f}%')
