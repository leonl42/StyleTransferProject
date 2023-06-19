import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import ToTensor,ToPILImage,Normalize
import os
from tqdm import tqdm
from PIL import Image
from utils.utility import normalize_cw
import pickle

class Parallel(nn.Module):
    """
    Layer that applies a given layer to the input and furthermore extracts and propagates feature values through the network.
    """
    def __init__(self, layer, content, style) -> None:
        super().__init__()
        self.layer = layer
        self.content = content
        self.style = style

    def forward(self,x):
        # Input x is a tuple: (<torch.Tensor>, <list>, <list>)
        img,content_list,style_list = x

        img = self.layer(img)
        if self.content:
            content_list.append(img)
        if self.style:
            style_list.append(img)

        return (img,content_list,style_list)

def construct_style_loss_model(model, content_layers, style_layers):
    """
    Given a pytorch sequential model, construct a new model that returns the features values of specified layers.

    Args:
        - model <nn.Sequential>: pytorch model from whose layers we want to use for feature extraction
        - content_layers <list<int>>: indeces of the layers from which we want to extract the features for the "content"
        - style_layers <list<int>>: indeces of the layers from which we want to extract the features for the "style"
    """
    seq = nn.Sequential()
    print(range(max(content_layers+style_layers)+1))
    for i in range(max(content_layers+style_layers)+1):
        layer = model[i]
        seq.add_module("Model layer: " + str(i) + " | Content layer: " + str(i in content_layers) + " | Style layer: " + str(i in style_layers),Parallel(layer,i in content_layers, i in style_layers))
    return seq


class Deconv(nn.Module):
    """
    Deconv layer that applies a layer and after that interpolates the output into the specified target shape
    """
    def __init__(self, layer, target_shape) -> None:
        super().__init__()
        self.layer = layer
        self.target_shape = target_shape

    def forward(self, x):
        x = self.layer(x)
        if not x.size() == self.target_shape:
            x = F.interpolate(x,self.target_shape, mode = "bilinear")

        return x
    

def construct_decoder_from_encoder(model, channels, input_width, input_height):
    """
    Construct a matching decoder to the given encoder by applying a mix of ConvTranspose2D and interpolation

    Args:
        - model <nn.Sequential>: pytorch model which represents the decoder
        - channels <int>: Number of channels of the input image
        - input_width <int>: pixel width of the input image
        - input_width <int>: pixel height of the input image
    """

    example_input = torch.rand((1,channels,input_width,input_height))
    decoder = []
    
    for i in range(len(model)):

        layer = model[i]
        example_output = layer(example_input)

        if isinstance(layer, nn.Conv2d):
            conv2d_transpose = nn.ConvTranspose2d(out_channels=example_input.size(1),in_channels=example_output.size(1),kernel_size=layer.kernel_size,stride=layer.stride)
            decoder.append(Deconv(conv2d_transpose,example_input.size()[2:]))
            decoder.append(nn.ReLU())

        example_input = example_output

    # remove last activation function
    decoder = decoder[:-1]

    # add sigmoid as last activation function
    decoder.insert(0,nn.Sigmoid())

    return nn.Sequential(*reversed(decoder))


class AdaIN(nn.Module):
    """
    Implementation of the Adaptive Instance Normalization layer proposed by Huang & Belongie (2017)
    """


    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, f, f_style):

        # f has size (batch_size x C x W x H)
        # f_style has size (batch_size x C x W x H)

        # get statistics parameters of activations across channels
        # (batch_size x C x 1 x 1)
        mean_f = f.mean((2,3),keepdim = True)
        std_f = f.std((2,3),keepdim = True)
        mean_f_style = f_style.mean((2,3),keepdim = True)
        std_f_style = f_style.std((2,3),keepdim = True)

        # normalizes the activations of our image and aligns them with the distribution of activations of the style image
        return std_f_style*((f - mean_f)/(std_f+1e-7)) + mean_f_style
    

class SANet(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=1,stride=1)
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=1,stride=1)
        self.conv3 = nn.Conv2d(channels,channels,kernel_size=1,stride=1)

    def forward(self, f_c, f_s):
        """
        f_c has shape (Batch_size x channels x width x height)
        f_s has shape (Batch_size x channels x width x height)
        """

        # perform channels wise normalization
        f_c_norm, f_s_norm = normalize_cw(f_c, f_s)

        b,c,w,h = f_c.size()

        f_c = self.conv1(f_c_norm).view(b,c,-1)
        g_s = self.conv2(f_s_norm).view(b,c,-1)
        h_s = self.conv3(f_s).view(b,c,-1)

        attention_map = torch.bmm(g_s,torch.transpose(f_c,1,2))
        
        return torch.bmm(attention_map,h_s).view(b,c,w,h)
    
class SAStyleNetwork(nn.Module):
    def __init__(self, feature_shape, feature_style_shape, decoder) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(feature_shape[0],feature_shape[0],kernel_size=1,stride=1)
        self.conv2 = nn.Conv2d(feature_style_shape[0],feature_style_shape[0],kernel_size=1,stride=1)
        self.conv3 = nn.Conv2d(max([feature_shape[0],feature_style_shape[0]]),max([feature_shape[0],feature_style_shape[0]]),kernel_size=3, stride=3)

    def forward(self, img, img_style):
        pass