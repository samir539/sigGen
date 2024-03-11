import torch.nn as nn 
import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from mixer import MLPMixer
import torch.optim as optim

#import MNIST and load into dataloader
transform = transforms.Compose([
    transforms.ToTensor(),  # This will handle conversion from PIL to PyTorch tensors
])
train_set = datasets.MNIST(root='./data', train=True, download=True,transform=transform)
train_loader = DataLoader(train_set,batch_size=16, shuffle=True)

test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(train_set,batch_size=16, shuffle=False)

#model params
data_iter = iter(train_loader)
images,label = data_iter.next()
first_img = images[0]
first_label = label[0]
image_height, image_width = first_img.shape[-2], first_img.shape[-1]
channels = first_img.shape[0]
channel_dim = 21 #hidden val
patch_dim = 8
patch_mlp_dim = 32
channel_mlp_dim = 32
mixer_layer_num = 2
class_num = 10

learning_rate = 0.01


#init model
MLP_mixer_model = MLPMixer(image_width,image_height,channels,channel_dim,patch_dim,patch_mlp_dim,channel_mlp_dim,mixer_layer_num,class_num)
print(repr(MLP_mixer_model))

#loss 
loss_criterion = nn.CrossEntropyLoss()
#optimiser 
optimiser = optim.Adam(MLP_mixer_model.parameters(),lr=learning_rate)

#CPU GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MLP_mixer_model.to(device)

epoch_num = 5

 

#Write training loop

# for every batch in MNIST  do a forward pass and get predition 
# compare prediction to true label and compute LOSS
# update grads 
# repeat until converged


# Call training loop

#test model inference