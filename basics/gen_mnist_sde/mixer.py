import torch.nn as nn 


class MixerMLP(nn.Module):
    """
    class to implement a MLP block found twice in a single mixer layer

    Class params:
    -------
    dim: the dim of the input (same as output)
    layer_width: width of a single layer


    Attributes:
    -------
    linear1: first linear layer
    linear2: second linear layer
    GELU: non-linear activation

    Methods:
    -------
    forward: a forward pass of the network


    """
    def __init__(self,dim,layer_width):
        super().__init__()
        dim = dim
        layer_width = layer_width
        self.linear1 = nn.linear(dim,layer_width)
        self.linear2 = nn.linear(layer_width,dim)

    def __repr__(self):
        return "mixer block"
    
    
    
    



    




class MixerBlock():
    pass

class MixerFull():
    pass
