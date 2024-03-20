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
    :param data: a data sample
    :param int_beta: function int_beta
    :param t: the time t to compute the sample value from p(x(t)|x(0))
    :return mean: the mean of the perturbation kernel
    :return std: return the standard deviation of the perturbation kernel 
    """
    mean = data*np.exp(-0.5*int_beta(t))
    std = 1 - np.exp(-int_beta(t))
    return mean, std




#loss function
def loss_function(data,int_beta,weight,score_model):
    """
    Function to compute the loss 
    :param data: a batch of the data dim:[n_sample, height, width]
    """
    t = torch.rand(data.shape[0])
    mean,std = perturbation_kernel(data,int_beta,t)
    #sample t
    #perturbed x val
    #score
    #
    


#batch loss function


#load mnist

#make step

#training loop
