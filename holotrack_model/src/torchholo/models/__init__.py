# coding: utf-8

# External imports

# Local imports
from .holosolver import *


def build_model(cfg, hologram):
    physical_params = cfg["physical_params"]
    nerf_params = cfg["nerf_params"]
    return eval(f"{cfg['class']}(physical_params, nerf_params, hologram)")
