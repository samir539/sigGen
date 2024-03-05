import torch.nn as nn 
import torch 
import torchvision 
import numpy as np
import matplotlib.pyplot as plt

## generate test vector which represents an image ##
test_image = np.random.rand(64,64)
test_image.shape = (-1,1)


## MLP MIXER ##

class MLPBlock(nn.Module):
    """
    class to implement the MLPBlock which is used twice in the overall mixer when mixing across patches and channels

    class params: 
    -------------
    dim = dimension of the input data
    layer_width = the width of the linear layers (set as hyperparameter)

    attributes:
    -----------
    linear1 = the first fully connected layer of the MLP 
    linear2 = the second fully connected layer of the MLP
    activation = the GELU non-linear activation function 

    methods:
    --------
    forward: carry out a forward pass of the network  

    """
    def __init__(self,dim,layer_width):
        super().__init__()
        dim = dim 
        layer_width = layer_width
        self.linear1 = nn.linear(dim,layer_width)
        self.linear2 = nn.linear(layer_width,dim)
        self.activation = nn.GELU()

    def forward(self,X):
        """
        forward pass of the MLP block
        :param X: input data
        :return X: the output data after a forward pass of the MLP
        """
        X = self.linear1(X)
        X = self.activation(X)
        X = self.linear2(X)
        return X


class MixerLayer():
    pass
