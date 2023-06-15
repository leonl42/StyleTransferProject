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


class StyleTransferDataset(torch.utils.data.Dataset):
    def __init__(self, path, file_content_images, file_style_images, style_model, device):

        self.content_save_path = path + "content/"
        self.style_save_path = path + "style/"

        os.makedirs(self.content_save_path,exist_ok=True)
        os.makedirs(self.style_save_path,exist_ok=True)

        for i,file in enumerate(file_content_images):
            img = ToTensor()(Image.open(file)).to(device)
            _,content_features,_ = style_model(img.unsqueeze(0))
            with open(self.content_save_path + str(i) + ".pkl") as f:
                pickle.dump((img,content_features),f)

        for i,file in enumerate(file_style_images):
            img = ToTensor()(Image.open(file)).to(device)
            _,_,style_features = style_model(img.unsqueeze(0))
            with open(self.style_save_path + str(i) + ".pkl") as f:
                pickle.dump((img,style_features),f)

        self.content_ids = range(len(file_content_images))
        self.style_ids = range(len(file_style_images))

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        random_content_id = random.choice(self.content_ids)
        random_style_id = random.choice(self.style_ids)

        with open(self.content_save_path + str(random_content_id) + ".pkl") as f:
            content_img, content_features = pickle.load(f)
        with open(self.style_save_path + str(random_style_id) + ".pkl") as f:
            style_img, style_features = pickle.load(f)

        return content_img,content_features,style_img,style_features