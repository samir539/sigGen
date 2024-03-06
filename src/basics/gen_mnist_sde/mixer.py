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


class MixerLayer(nn.Module):
    """
    class to implement the Mixer layer of the MLPMixer
    
    class parameters:
    ----------------


    attributes:
    ----------
    layer_norm 
    MLP1 = instance of the mixer block class which represents MLP1
    MLP2 = instance of the mixer block class which reprsents MLP2


    methods:
    -------
    forward: run forward pass of MLPMixer Layer
    """
    def __init__(self,MLP1, MLP2):
        super().__init__()
        self.layernorm = nn.layerNorm()
        self.MLP1 = MLP1
        self.MLP2 = MLP2

    def forward(self,X):
        """
        Input data
        """
        X = self.MLP1.forward(self.layernorm(X)) + X
        X = self.MLP2.forward(self.layernorm(X.T)) + X.T
        return X


class MLPMixer(nn.Module):
    """
    class to implement full MLP mixer module 
    
    class parameters:
    ---------------

    attributes:
    ----------
    embedding_dim = the dimension of the latent space
    embedding_layer  = embedding layer used to embed patches into a latent space
    mixer_layer_num = the number of mixer layers in the model 
    global_pooling  = global pooling layer
    output_fc = fully connected output layer
    X = input data in form (n_image, channels, patches)


    methods:
    --------
    forward: run a forward pass of the full module 
    """

    def __init__(self, embedding_dim, mixer_layer_num):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.linear()
        self.mixer_layer_num = mixer_layer_num
        
        #embedding 

        #



#embedding functionality 
#patch generation and handling 
    
## make model in main ##
#handle MNIST
#global pooling and fully connected 
#training loop 

