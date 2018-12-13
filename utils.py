import cv2
import numpy as np

def bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def flipimg(image):
    return cv2.flip(image, 1)

def cropimg(image):
    cropped = image[60:130, :]
    return cropped

def resize(image, shape=(160, 70)):
    return cv2.resize(image, shape)

def crop_and_resize(image):
    cropped = cropimg(image)
    resized = resize(cropped)
    return resized