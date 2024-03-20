import torch.nn as nn
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import numpy as np
from mixer import MLPMixer

#######
##SDE##
#######
#dy(t) = -0.5beta(t)y(t)dt+sqrt(beta(t))dw(t)

#perturb_kernel
def perturbation_kernel(data,int_beta,t):
    """
    Compute the mean and standard deviation of the perturbation kernel given a t val
    :param data: a mini-batch of the data dim:[n_sample, channel, height,width]
    :param int_beta: function int_beta
    :param t: the time t to compute the sample value from p(x(t)|x(0))
    :return mean: the mean of the perturbation kernel
    :return std: return the standard deviation of the perturbation kernel 
    """
    mean = data*np.exp(-0.5*int_beta(t)) #dim [n_sample, channel, height, width]
    std = 1 - np.exp(-int_beta(t))
    return mean, std




#loss function
def loss_function(data,int_beta,weight,score_model):
    """
    Function to compute the loss 
    :param data: a mini-batch of the data dim:[n_sample, channel,height, width]
    """
    t = torch.rand(data.shape[0])
    mean,std = perturbation_kernel(data,int_beta,t)
    noise = torch.randn_like(data) #dim [n_sample, channel, height, width]
    perturbed_x = data+ noise*mean + std[:,None,None,None] #dim [n_sample, channel, height, width]
    score = score_model(perturbed_x,t) #dim [n_sample, channel,height,width]
    loss = torch.mean(torch.sum((score + noise/std[:,None,None,None])**2, dim=(1,2,3))) #scalar


    


#batch loss function


#load mnist

#make step

#training loop
