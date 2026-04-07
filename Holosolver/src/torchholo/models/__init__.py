# coding: utf-8

# External imports

# Local imports
from .holosolver import *


def build_model(cfg, hologram, device):
    physical_params = cfg["physical_params"]
    nerf_params = cfg["nerf_params"]
    regularization_params = cfg["regularization"]
    pre_training_params = cfg["pre_training"]
    vram_params = cfg["vram_params"]
    return eval(f"{cfg['class']}(physical_params, nerf_params, regularization_params, pre_training_params, vram_params, hologram, device)")
