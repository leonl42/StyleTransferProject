import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import ToTensor,ToPILImage
import os
from tqdm import tqdm
from PIL import Image
import random
import pickle
import math
import torch.utils.data as data
from torchvision.transforms import ToTensor


class LoadFilesDataset(torch.utils.data.Dataset):
    """
    Pytorch dataset that loads a random file from a directory. Since this is a random process, it is enough if the dataset has length "Batch size".
    Note that the files have to be labeled 0.jpg ... N.jpg where N is the last file.
    """

    def __init__(self, file_path, batch_size):
        self.file_path = file_path
        self.batch_size = batch_size
        self.num_files = len(os.listdir(file_path))
        self.to_tensor = ToTensor()
      

    def __len__(self):
        return self.batch_size

    def __getitem__(self, _):

        random_file_id = str(random.randint(0,self.num_files-1))

        # Since PIL has the format [W x H x C], and ToTensor() transforms it into [C x H x W], we have to permute the tensor to shape [C x W x H]
        img = self.to_tensor(Image.open(self.file_path + random_file_id + ".jpg")).permute(0,2,1)

        return img
    
