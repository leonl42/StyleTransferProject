import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def content_gatyes(f,f_target):
    """
    Computes the content loss proposed by Gatyes et al. (2016)
    Args:
        - f <torch.Tensor>: feature tensor of the image we are optimizing of shape (batch_size, width, height, channel)
        - f_target <torch.Tensor>: feature tensor of the content image with shape (batch_size, width, height, channel)
    """

    # reshape the feature maps into shape (batch_size x M_l x channels), where M_l is height*width of our feature map
    f, f_target = torch.flatten(f,start_dim=1,end_dim=2), torch.flatten(f_target,start_dim=1,end_dim=2)

    return F.mse_loss(f, f_target)

def style_gatyes(f, f_target):
    """
    Computes the style loss proposed by Gatyes et al. (2016)
    Args:
        - f <torch.Tensor>: feature tensor of the image we are optimizing of shape (batch_size, width, height, channel)
        - f_target <torch.Tensor>: feature tensor of the style image with shape (batch_size, width, height, channel)
    """

    # reshape the feature maps into shape (batch_size x M_l x channels), where M_l is height*width of our feature map
    f, f_target = torch.flatten(f,start_dim=1,end_dim=2), torch.flatten(f_target,start_dim=1,end_dim=2)

def style_mmd(f, f_target, kernel):

    # The number of rows we are going to sample
    sample_size = f.size(1)
    
    # indexing tensor for sampling feature columns of size (sample_size, )
    random_indices_f = torch.randint(low=0,high=sample_size,size=sample_size)
    random_indices_f_target = torch.randint(low=0,high=sample_size,size=sample_size)


def style_mmd_polynomial(f, f_target):
    """
    Computes the style loss proposed by Gatyes et al. (2016) by directly minimizing the Maximum Mean Discrepancy
    between the feature distributions using a polynomial kernel(Li et al. (2017))
    Args:
        - f <torch.Tensor>: feature tensor of the image we are optimizing of shape (batch_size, width, height, channel)
        - f_target <torch.Tensor>: feature tensor of the style image with shape (batch_size, width, height, channel)
    """

    # reshape the feature maps into shape (batch_size x M_l x channels), where M_l is height*width of our feature map
    f, f_target = torch.flatten(f,start_dim=1,end_dim=2), torch.flatten(f_target,start_dim=1,end_dim=2)

    # The number of rows we are going to sample
    sample_size = f.size(1)*f.size(2)

    