import cv2
from PIL import Image
from torchvision.transforms import ToTensor
import os
import torch

def normalize(f, f_target):
    """
    Normalizes the features globally within a batch sample. Note that while Wang et al. (2021) propose
    a softmax normalization, they also say that any form of normalization yields good results. Due to the fact that softmax did not really work for me 
    (probably due to the now small feature values paired with ill hyperparams) I decided to use a simple 0-mean 1-std normalization, which seems to be more stable
    and also improves style quality.
    Args:
        - f <torch.Tensor>: feature tensor of the image we are optimizing of shape (batch_size, channels, width, height)
        - f_target <torch.Tensor>: feature tensor of the style image with shape (batch_size, channels, width, height)
    """

    # normalize feature tensors
    f , f_target = (f - f.mean((1,2,3),keepdim=True))/f.std((1,2,3),keepdim=True),(f_target - f_target.mean((1,2,3),keepdim=True))/f_target.std((1,2,3),keepdim=True)

    return f,f_target

def normalize_cw(f, f_target):
    """
    Applies a 0-mean 1-variance normalization channel wise.
    Args:
        - f <torch.Tensor>: feature tensor of the image we are optimizing of shape (batch_size, channels, width, height)
        - f_target <torch.Tensor>: feature tensor of the style image with shape (batch_size, channels, width, height)
    """

    # normalize feature tensors
    f , f_target = (f - f.mean((2,3),keepdim=True))/f.std((2,3),keepdim=True),(f_target - f_target.mean((2,3),keepdim=True))/f_target.std((2,3),keepdim=True)

    return f,f_target


def reshape(f,f_target):
    """
    Reshape features from (batch_size, channels, width, height) to (batch_size, channels, width*height) = (batch_size, channels, M_l)
    Args:
        - f <torch.Tensor>: feature tensor of the image we are optimizing of shape (batch_size, channels, width, height)
        - f_target <torch.Tensor>: feature tensor of the style image with shape (batch_size, channels, width, height)
    """

    # reshape the feature maps into shape (batch_size x channels x M_l ), where M_l is height*width of our feature map
    f, f_target = torch.flatten(f,start_dim=2,end_dim=3), torch.flatten(f_target,start_dim=2,end_dim=3)

    return f, f_target

def video_to_frame_generator(file, img_size):
    """
    Loads a video from the specified path and returns a generator that yields each frame of the video
    Args:
        - file <String> file path of the video file
        - img_size <(int, int)> tuple of int's representing image width and image height
    """

    vid = cv2.VideoCapture(file)

    while True:

        exists,image = vid.read()

        if not exists:
            break
        image = cv2.resize(image, img_size, interpolation = cv2.INTER_AREA)
        yield Image.fromarray(image)

def video_to_frames(frame_generator, save_path):
    """
    Save each element in frame_generator to the specified path
    Args:
        - frame_generator <generator>: Generator that yields elements
        - save_path <String>: Where to save the elements
    """

    os.makedirs(save_path, exist_ok=True)

    for i,frame in enumerate(frame_generator):
        frame.save(save_path + str(i) + ".jpg")

