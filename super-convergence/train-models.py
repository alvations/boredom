import time
import os
from glob import glob

import numpy as np

import sys

from tqdm import tqdm

import torch
from torch import nn
import torch.optim as optim

from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from CLR_preview import CyclicLR
from adamW import AdamW


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
jitter_param = 0.4
lighting_param = 0.1

# Input pre-processing for train data
preprocess = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=jitter_param, contrast=jitter_param,
                                     saturation=jitter_param),
        transforms.ToTensor(),
        normalize
    ])

# No transformation.
#dataset = ImageFolder(root="/disk2/rdlc_data/train/", transform=ToTensor())

netname, batchsz = sys.argv[1], sys.argv[2]
#netname, batchsz = 'densenet', 600

dataset = ImageFolder(root="train/", transform=preprocess)
dataloader = DataLoader(dataset, batch_size=int(batchsz), shuffle=True, pin_memory=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

pre_nets = {'vgg16': models.vgg16(pretrained=True),
 'resnet18': models.resnet18(pretrained=True),
 'alexnet': models.alexnet(pretrained=True),
 'squeezenet': models.squeezenet1_1(pretrained=True),
 'densenet': models.densenet161(pretrained=True),
 'inception': models.inception_v3(pretrained=True)
 
 }


net = pre_nets[netname].train()
net = nn.DataParallel(net, device_ids=[0,1]).to(device)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

optimizer = AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay = 0.1)

num_classes = 43
clr_stepsize = (num_classes*50//int(batchsz))*4
clr_wrapper = CyclicLR(optimizer, step_size=clr_stepsize)

for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader)):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        clr_wrapper.batch_step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0 and i != 0:    # print every 2000 mini-batches
            lrs = [p['lr'] for p in optimizer.param_groups]
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100), lrs)
            running_loss = 0.0
    torch.save(net.state_dict(), 'models/{}-{}-{}.pth'.format(netname, batchsz, epoch))

