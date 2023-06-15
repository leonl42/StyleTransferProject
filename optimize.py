import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm

def optimize(content_image, style_images, model, content_layers, style_layers, device, path = None):
    """
    Performs the style transfer by directly optimizing the image and returns the content image with the style of the style images

    Args:
        - content_image <torch.Tensor>: pytorch tensor of size (in_channels, width, height)
        - style_images <list<torch.Tensor>> list of tensors of size (in_channels, width, height).
            All tensors have to match the size of the content image
        - model <torch.nn.Sequential>: pytorch Sequential model
        - content_layers <list<int>>: list of layer indexes 
        - style_layers <list<int>>: list of layer indexes
        - device <torch.device>: On which device to perform the optimization (cpu or gpu)
        - path <String>: If specified and device is gpu, the layer activations will not be kept in memory but stored in files at the given location
    """

    MODE = "on_device"
    if device == torch.device("cpu") and path is not None:
        MODE = "on_storage"
        os.makedirs(path + "content/",exist_ok=True)
        os.makedirs(path + "style/",exist_ok=True)
        for i in content_layers:
            os.makedirs(path + "content/" + str(i) + "/",exist_ok=True)
        for i in style_layers:
            os.makedirs(path + "style/" + str(i) + "/",exist_ok=True)
    
    
    # (1, in_channels, width, height): expand by batch size
    content_image = content_image.unsqueeze(0)

    # (num_style_images, in_channels, width, height): convert list of style images to tensor
    style_images = torch.stack(style_images,dim=0)


    # set the image we are going to optimize to the content image
    result_img = content_image.clone()

    for i in range(max(content_layers+style_layers)+1):
        content_image = model[i](content_image)
        style_images = model[i](style_images)
        
        if i in content_layers:
            pass

        if i in style_layers:
            pass