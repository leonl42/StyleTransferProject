import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def content_gatyes(f,f_target):
    """
    Computes the content loss proposed by Gatyes et al. (2016)
    Args:
        - f <torch.Tensor>: feature tensor of the image we are optimizing of shape (batch_size, channels, width, height)
        - f_target <torch.Tensor>: feature tensor of the style image with shape (batch_size, channels, width, height)
    """

    #f, f_target = torch.exp(f)/torch.exp(f).sum((1,2,3),keepdim=True), torch.exp(f_target)/torch.exp(f_target).sum((1,2,3),keepdim=True)

    # reshape the feature maps into shape (batch_size x channels x M_l ), where M_l is height*width of our feature map
    f, f_target = torch.flatten(f,start_dim=2,end_dim=3), torch.flatten(f_target,start_dim=2,end_dim=3)

    return ((f - f_target)**2).mean((1,2))

def style_gatyes(f, f_target):
    """
    Computes the style loss proposed by Gatyes et al. (2016)
    Args:
        - f <torch.Tensor>: feature tensor of the image we are optimizing of shape (batch_size, channels, width, height)
        - f_target <torch.Tensor>: feature tensor of the style image with shape (batch_size, channels, width, height)
    """

    #f, f_target = torch.exp(f)/torch.exp(f).sum((1,2,3),keepdim=True), torch.exp(f_target)/torch.exp(f_target).sum((1,2,3),keepdim=True)

    # reshape the feature maps into shape (batch_size x channels x M_l ), where M_l is height*width of our feature map
    f, f_target = torch.flatten(f,start_dim=2,end_dim=3), torch.flatten(f_target,start_dim=2,end_dim=3)

    G = torch.bmm(f, torch.transpose(f,1,2))

    #G.div(f.size(0)*f.size(1))

    A = torch.bmm(f_target, torch.transpose(f_target,1,2))
    #A.div(f_target.size(0)*f_target.size(1))
    return ((G - A)**2).mean((1,2))


def style_mmd_polynomial(f, f_target):
    """
    Computes the style loss proposed by Gatyes et al. (2016) by directly minimizing the Maximum Mean Discrepancy
    between the feature distributions using a polynomial kernel(Li et al. (2017))
    Args:
        - f <torch.Tensor>: feature tensor of the image we are optimizing of shape (batch_size, channels, width, height)
        - f_target <torch.Tensor>: feature tensor of the style image with shape (batch_size, channels, width, height)
    """

    # reshape the feature maps into shape (batch_size x channels x M_l ), where M_l is height*width of our feature map
    f, f_target = torch.flatten(f,start_dim=2,end_dim=3), torch.flatten(f_target,start_dim=2,end_dim=3)

    # The number of rows we are going to sample
    sample_size = 1024*8
    channels = f.size(1)
    batch_size = f.size(0)

    # indexing tensor for sampling feature columns of size (batch_size x channels x sample_size)
    random_indices_f = torch.randint(low=0,high=f.size(2),size=(sample_size,)).to(f.device).unsqueeze(0).unsqueeze(0).repeat(batch_size,channels,1)
    random_indices_f_target = torch.randint(low=0,high=f.size(2),size=(sample_size,)).to(f.device).unsqueeze(0).unsqueeze(0).repeat(batch_size,channels,1)

    # randomly sampled column vectors (batch_size x sample_size x channels)
    random_columns_f = torch.gather(f,-1,random_indices_f).permute(0,2,1)
    random_columns_f_target = torch.gather(f_target,-1,random_indices_f_target).permute(0,2,1)

    pol_f = torch.bmm(random_columns_f,torch.transpose(random_columns_f,1,2))**2
    pol_s = torch.bmm(random_columns_f_target,torch.transpose(random_columns_f_target,1,2))**2
    pol_f_s = torch.bmm(random_columns_f,torch.transpose(random_columns_f_target,1,2))**2

    loss = (pol_f + pol_s - 2*pol_f_s).sum((1,2))/(4*sample_size**2 * channels**2)

    return loss

def adaIN(f, f_target): 
    mean = f.mean((2,3))
    mean_target = f_target.mean((2,3))
    std = torch.std(f,dim=(2,3))
    std_target = torch.std(f_target,dim=(2,3))

    return ((mean - mean_target)**2 + (std - std_target)**2).sum(1)/f.size(1)
    