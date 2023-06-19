import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import ToTensor,ToPILImage
import os
from tqdm import tqdm
from PIL import Image,ImageOps
import random
import pickle
import math
import torch.utils.data as data
from torchvision.transforms import ToTensor


class LoadCoCoDataset(torch.utils.data.Dataset):
    """
    Pytorch dataset that loads a random file from a directory. 
    """

    def __init__(self, file_path, batch_size, image_size):
        self.file_path = file_path
        self.batch_size = batch_size
        self.files = os.listdir(file_path)
        self.num_files = len(self.files)
        self.to_tensor = ToTensor()
        self.image_size = image_size
      

    def __len__(self):
        return self.batch_size

    def __getitem__(self, _):

        random_file = random.choice(self.files)
        img = Image.open(self.file_path + random_file).convert("RGB")

        # center crop image
        img = ImageOps.fit(img,(min(img.size),min(img.size)))

        img = img.resize(self.image_size)
        
        # Since PIL has the format [W x H x C], and ToTensor() transforms it into [C x H x W], we have to permute the tensor to shape [C x W x H]
        img = self.to_tensor(img).permute(0,2,1)

        return img
    
