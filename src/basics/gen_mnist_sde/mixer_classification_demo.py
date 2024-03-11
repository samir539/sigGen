import torch.nn as nn 
import torch 
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from mixer import MLPMixer

#import MNIST
train_set = datasets.MNIST(root='./data', train=True, download=True)
train_loader = DataLoader(train_set,batch_size=16, shuffle=True)

test_set = datasets.MNIST(root='./data', train=False, download=True)
test_loader = DataLoader(train_set,batch_size=16, shuffle=False)



#load into dataloader 

#Write training loop

# Call training loop

#test model inference