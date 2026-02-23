# coding: utf-8

# External imports

# Local imports
from .holosolver import *


def build_model(cfg, hologram, device):
    physical_params = cfg["physical_params"]
    nerf_params = cfg["nerf_params"]
    regularization_params = cfg["regularization"]
    return eval(f"{cfg['class']}(physical_params, nerf_params, regularization_params, hologram, device)")
