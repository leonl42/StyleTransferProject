import cv2
from PIL import Image


def video_to_generator(file):
    """
    Loads a video from the specified path and returns a generator that yields each frame of the video
    """

    vid = cv2.VideoCapture(file)

    while True:

        exists,image = vid.read()

        if not exists:
            break

        yield Image.fromarray(image)

