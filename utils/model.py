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

def construct_style_model(model, content_layers, style_layers):
    #Normalize(mean = (0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    seq = nn.Sequential()
    for i in range(max(content_layers+style_layers)+1):
        layer = model[i]
        seq.add_module(Parallel(layer,i in content_layers, i in style_layers))
    return seq