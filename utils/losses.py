import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.utility import reshape


def content_gatyes(f,f_target):
    """
    Computes the content loss proposed by Gatyes et al. (2016)
    Args:
        - f <torch.Tensor>: feature tensor of the image we are optimizing of shape (batch_size, channels, width, height)
        - f_target <torch.Tensor>: feature tensor of the style image with shape (batch_size, channels, width, height)
    """

    f , f_target = reshape(f, f_target)

    return ((f - f_target)**2).mean((1,2))

def style_gatyes(f, f_target):
    """
    Computes the style loss proposed by Gatyes et al. (2016)
    Args:
        - f <torch.Tensor>: feature tensor of the image we are optimizing of shape (batch_size, channels, width, height)
        - f_target <torch.Tensor>: feature tensor of the style image with shape (batch_size, channels, width, height)
    """

    f , f_target = reshape(f, f_target)

    G = torch.bmm(f, torch.transpose(f,1,2))/(f.size(1)*f.size(2))

    A = torch.bmm(f_target, torch.transpose(f_target,1,2))/(f.size(1)*f.size(2))

    return ((G - A)**2).mean((1,2))


def style_mmd_polynomial(f, f_target):
    """
    Computes the style loss proposed by Gatyes et al. (2016) by directly minimizing the Maximum Mean Discrepancy (Gretton et al. (2012))
    between the feature distributions using a polynomial kernel (Li et al. (2017))
    Args:
        - f <torch.Tensor>: feature tensor of the image we are optimizing of shape (batch_size, channels, width, height)
        - f_target <torch.Tensor>: feature tensor of the style image with shape (batch_size, channels, width, height)
    """

    f , f_target = reshape(f, f_target)

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

    loss = (pol_f + pol_s - 2*pol_f_s).sum((1,2))

    return loss

def style_mmd_gaussian(f, f_target):
    """ 
    Minimizes the Maximum Mean Discrepancy (Gretton et al. (2012)) between the feature distributions using a gaussian kernel (Li et al. (2017))
    Args:
        - f <torch.Tensor>: feature tensor of the image we are optimizing of shape (batch_size, channels, width, height)
        - f_target <torch.Tensor>: feature tensor of the style image with shape (batch_size, channels, width, height)
    """

    f , f_target = reshape(f, f_target)

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

    SIGMA = 0.5

    dist_f = torch.exp(-torch.cdist(random_columns_f,random_columns_f)**2/(2*SIGMA**2))
    dist_s = torch.exp(-torch.cdist(random_columns_f_target,random_columns_f_target)**2/(2*SIGMA**2))
    dist_f_s = torch.exp(-torch.cdist(random_columns_f,random_columns_f_target)**2/(2*SIGMA**2))

    loss = (dist_f + dist_s - 2*dist_f_s).sum((1,2))

    return loss

def adaIN(f, f_target): 
    """
    Implementation of the AdaIN loss proposed by Li et al. (2017) (Here it is named Batch Normalization loss) and Huang & Belongie (2017)

    Args:
        - f <torch.Tensor>: feature tensor of the image we are optimizing of shape (batch_size, channels, width, height)
        - f_target <torch.Tensor>: feature tensor of the style image with shape (batch_size, channels, width, height)
    
    """

    f , f_target = reshape(f, f_target)

    mean = f.mean(2)
    mean_target = f_target.mean(2)
    std = f.std(2)
    std_target = f_target.std(2)
    return ((mean - mean_target)**2 + (std - std_target)**2).sum(1)/f.size(1)
    