import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import ToTensor,ToPILImage,Normalize
import os
from tqdm import tqdm
from PIL import Image
import pickle

class Parallel(nn.Module):
    def __init__(self, layer, content, style) -> None:
        super().__init__()
        self.layer = layer
        self.content = content
        self.style = style

    def forward(self,x):
        img,content_list,style_list = x

        img = self.layer(img)
        if self.content:
            content_list.append(img)
        if self.style:
            style_list.append(img)

        return (img,content_list,style_list)

def construct_style_loss_model(model, content_layers, style_layers):

    seq = nn.Sequential()
    print(range(max(content_layers+style_layers)+1))
    for i in range(max(content_layers+style_layers)+1):
        layer = model[i]
        seq.add_module("Model layer: " + str(i) + " | Content layer: " + str(i in content_layers) + " | Style layer: " + str(i in style_layers),Parallel(layer,i in content_layers, i in style_layers))
    return seq


class Deconv(nn.Module):
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
    example_input = torch.rand((1,channels,input_width,input_height))
    decoder = []

    for i in range(len(model)):
        layer = model[i]
        example_output = layer(example_input)

        if isinstance(layer, nn.Conv2d):
            conv2d_transpose = nn.ConvTranspose2d(out_channels=example_input.size(1),in_channels=example_output.size(1),kernel_size=layer.kernel_size,stride=layer.stride)
            decoder.append(Deconv(conv2d_transpose,example_input.size()[2:]))
            decoder.append(nn.ReLU())
        if isinstance(layer,nn.AvgPool2d) or isinstance(layer,nn.MaxPool2d):
            pass

        example_input = example_output

    # remove last activation function
    decoder = decoder[:-1]

    # add sigmoid as last activation function
    decoder.insert(0,nn.Sigmoid())

    # Reverse list since we want to have the reverse order than the encoder
    return nn.Sequential(*reversed(decoder))

