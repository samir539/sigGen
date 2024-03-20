import torch.nn as nn 
import torch 
import torchvision 
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat

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
        self.linear1 = nn.Linear(dim,layer_width)
        self.linear2 = nn.Linear(layer_width,dim)
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
    layernorm1 = the layer norm for channel mixing 
    layernorm2 = the layer norm for patch/token mixing 
    MLP1 = instance of the mixer block class which represents MLP1 [patch/token mixer]
    MLP2 = instance of the mixer block class which reprsents MLP2  [channel mixer]


    methods:
    -------
    forward: run forward pass of MLPMixer Layer
    """
    def __init__(self, patch_number,patch_mlp_dim, channel_dim, channel_mlp_dim):
        """

        :param channel_dim: the dimensions of the channel (the space we embed each patch into) not to be confused with the channels of an image (eg RGB)
        :param patch_mlp_dim: the width of the patch/token mixing MLP

        :parma channel_mlp_dim: the width of the channel mixing MLP
        """
        super().__init__()
        self.layernorm1 = nn.LayerNorm(channel_dim)
        self.layernorm2 = nn.LayerNorm(channel_dim)
        self.MLP1 = MLPBlock(patch_number, patch_mlp_dim)
        self.MLP2 = MLPBlock(channel_dim, channel_mlp_dim)

    def forward(self,X):
        """
        Input data X with dim (n_samples, patches, channels)
        """
        # X has shape [n_samples, patches, channels]
        y = self.layernorm1(X)
        y = rearrange(y, 'n_samples patches channels -> n_samples channels patches')
        y = self.MLP1.forward(y)
        y = rearrange(y, 'n_samples channels patches -> n_samples patches channels')
        X = X + y #skip connection 
        y = self.layernorm2(X)
        X = X + self.MLP2.forward(y) #skip connection
        return X


class MLPMixer(nn.Module):
    """
    class to implement full MLP mixer module 
    
    class parameters:
    ---------------

    attributes:
    ----------
    num_patches = the number of patches 
    patch_dim = the dimensions of the patches
    embedding_layer  = embedding layer used to embed patches into a latent space
    mixerlayers = modulelist containing the n mixer layer blocks
    pre_head_norm = the layer norm before the classification head
    fully_connected_classifer_layer = the linear classifier layer #(replace with softmax?) 
    mixer_layer_num = the number of mixer layers in the model 


    methods:
    --------
    forward: run a forward pass of the full module 
    """

    def __init__(self,image_width,image_height,channels,channel_dim,patch_dim, patch_mlp_dim, channel_mlp_dim, mixer_layer_num,class_num,t1):
        """

        :param image_width: the width of the images
        :param image_height: the height of the images
        :param channels: the number of RGB channels in the image
        :param channel_dim: the dimension of the embedding layer 
        :param patch_dim: the dimensions of a patch
        :param patch_mlp_dim: the width of the patch mixing mlp (MLP1)
        :param channel_mlp_dim: the width of the channel mixing mlp (MLP2)
        :param mixer_layer_num: the number of mixer layer blocks
        :param class_num: the number of possible target classes
        :param t1: the value of t at the end of the diffusion (forward process) 
        """
        super().__init__()
        self.num_patches = (image_height*image_width)//(patch_dim**2)
        self.patch_dim = patch_dim
        self.embedding_layer = nn.Linear((channels+1)*patch_dim*patch_dim,channel_dim)
        self.mixerlayers = nn.ModuleList([MixerLayer(patch_number=self.num_patches,
                                                     patch_mlp_dim=patch_mlp_dim,
                                                     channel_dim=channel_dim, 
                                                     channel_mlp_dim= channel_mlp_dim) for i in range(mixer_layer_num)])
        self.pre_head_norm = nn.LayerNorm(channel_dim)
        self.fully_connected_classfier_layer = nn.Linear(channel_dim,class_num)
        self.t1 = t1

    def __repr__(self):
        return "this is an instance of mlp mixer model"
    
    def forward(self, X,t):
        """
        forward pass of the full mlp mixer
        :param t: the value for t (the score is computed at time t of the diffusion)
        :param X: input images of form (n_sample,channel,width, height)
        :return y: the class label outputs (1, n_classes)
        """

        #get time as fraction
        t = torch.tensor(t)
        t = t/self.t1
        t = repeat(t, "-> n_sample 1 w h", n_sample=X.shape[0], w=X.shape[2],h=X.shape[3])
        X = torch.cat((X,t),dim=1) # X dim [n_sample, c+1, width, height]
        #into patches
        X = rearrange(X, "n_sample c (w p1) (h p2) -> n_sample (w h) (c p1 p2)", p1=self.patch_dim, p2=self.patch_dim)
        #embedding 
        X = self.embedding_layer(X) # X dim [n_sample, n_patches, channels]
        # N mixer layers
        for layer in self.mixerlayers:
            X = layer(X) # X[n_sample, n_patches, channels]
        
        # pre-classification head layer norm
        X = self.pre_head_norm(X)
        # global average pooling
        X = X.mean(dim=1) #X dim [n_sample, channels]

        # class prediction with linear layer 
        y = self.fully_connected_classfier_layer(X) # X dim [n_samples, n_class]
        return y 



if __name__ == "__main__":
    test_image = torch.rand(1,3,64,64)
    image_height, image_width = test_image.shape[-2], test_image.shape[-1]
    channels = test_image.shape[1]
    channel_dim = 21 #hidden val
    patch_dim = 8
    patch_mlp_dim = 32
    channel_mlp_dim = 32
    mixer_layer_num = 2
    class_num = 10
    myMixer = MLPMixer(image_width,image_height,channels,channel_dim,patch_dim,patch_mlp_dim,channel_mlp_dim,mixer_layer_num,class_num,1)
    print(repr(myMixer))
    output = myMixer.forward(test_image,0.5)
    print(output.size())
    
    print("this is the output of a forward pass of mlp mixer", output)
    

    