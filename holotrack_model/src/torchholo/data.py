# coding: utf-8

# Standard imports
import logging
import random
import sys

# External imports
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision.transforms import v2
from PIL import Image
import numpy as np

#import matplotlib.pyplot as plt


def get_hologram(data_config):
    
    logging.info("  - Load holographic image")
    image_path = data_config["root_dir"]

    image = Image.open(image_path)

    logging.info("  - Apply transforms on holographic image")
    preprocess_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True) #Normalize (/255.0)
    ]
    
    transform = v2.Compose(preprocess_transforms)
    image_tensor = transform(image)
    image_tensor = image_tensor.squeeze(0)
    
    logging.info("  - Return holographic image")
    return image_tensor

def test_dataloaders() : 
    config = {
        "root_dir" : "../holograms/E_coli.tif",
    }
    hologram = get_hologram(config)
    print(f"Hologram shape (H, W) : {hologram.shape[0]}, {hologram.shape[1]}")
    
if __name__=="__main__" : 
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    test_dataloaders()
