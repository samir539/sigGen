import torch.nn as nn 
import torch 
import torchvision 
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

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
        :param X: input data input form (n_sample, channels, patches) or (n_sample, patches, channels)
        :return X: the output data after a forward pass of the MLP
        """
        X = self.linear1(X) # X dim (n_samples,*, layer_width)
        X = self.activation(X) # X dim (n_samples, *, layer_width)
        X = self.linear2(X) # X dim (n_samples, *, dim)
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
    def __init__(self, channel_dim, patch_mlp_dim, patch_number, channel_mlp_dim):
        super().__init__()
        self.layernorm1 = nn.layerNorm(channel_dim)
        self.layernorm2 = nn.layerNorm(channel_dim)
        self.MLP1 = MLPBlock(patch_number, patch_mlp_dim)
        self.MLP2 = MLPBlock(channel_dim, channel_mlp_dim)

    def forward(self,X):
        """
        Input data X with dim (n_samples, patches, channels)
        """
        # X has shape [n_samples, patches, channels]
        X = rearrange(X, 'n_samples patches channels -> n_samples channels patches')
        X = self.MLP1.forward(self.layernorm1(X)) + X
        X = rearrange(X, 'n_samples channels patches -> n_samples patches channels')
        X = self.MLP2.forward(self.layernorm2(X)) + X
        return X


class MLPMixer(nn.Module):
    """
    class to implement full MLP mixer module 
    
    class parameters:
    ---------------

    attributes:
    ----------
    embedding_dim = the dimension of the latent space
    image_width = the width of the images
    image_height = the height of the images
    num_patches = the number of patches 
    patch_dim = the dimensions of the patches
    embedding_layer  = embedding layer used to embed patches into a latent space
    mixer_layer_num = the number of mixer layers in the model 
    global_pooling  = global pooling layer
    output_fc = fully connected output layer
    X = input data in form (n_image, channels, width, height)


    methods:
    --------
    forward: run a forward pass of the full module 
    """

    def __init__(self,image_width,image_height,channels,channel_dim,patch_dim, patch_mlp_dim, channel_mlp_dim, mixer_layer_num,class_num):
        super().__init__()
        self.num_patches = (image_height*image_width)//(patch_dim**2)
        self.patch_dim = patch_dim
        self.embedding_layer = nn.linear(channels*patch_dim*patch_dim,channel_dim)
        self.mixerlayers = nn.ModuleList([MixerLayer(channel_dim=channel_dim,
                                                     patch_mlp_dim=patch_mlp_dim, 
                                                     patch_number=self.num_patches,
                                                     channel_mlp_dim= channel_mlp_dim) for i in range(mixer_layer_num)])
        self.fully_connected_classfier_layer = nn.linear(channel_dim,class_num)
    
    def forward(self, X):
        """
        forward pass of the full mlp mixer
        :param X: input images of form (n_sample,channel,width, height)
        :return y: the class label
        """


        #into patches
        X = rearrange(X, "n_sample c (w p1) (h p1) -> n_sample (w h) (c p1 p2)", p1=self.patch_dim, p2=self.patch_dim)
        #embedding 
        X = self.embedding_layer(X) # X dim [n_sample, n_patches, channels]
        X = rearrange(X, "n_sample n_patches channels -> n_sample channels n_patches")
        # N mixer layers
        for layer in self.mixerlayers:
            X = layer(X) # X[n_sample, n_patches, channels]
        
        # global average pooling
        X = X.mean(dim=1) #X dim [n_sample, channels]
        # class prediction with linear layer 
        y = self.fully_connected_classfier_layer(X) # X dim [n_samples, n_class]
        return y 






#embedding functionality 
#patch generation and handling 
    
## make model in main ##
#handle MNIST
#global pooling and fully connected 
#training loop 

