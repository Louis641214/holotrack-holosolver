# coding: utf-8

# External imports
import torch
import logging
import sys
import os
import shutil

from .morpholonet import MorpHoloNet
from .holosolver import HoloSolver
# Local imports
from . import build_model


# NOTE: not implemented yet
def test_morpholonet(cfg, device):
    logging.info(" 🛠️ Test MorpHoloNet\n")
    model = MorpHoloNet(cfg)
    model.to(device)
    logging.info(f"✅ Model called : {model}\n")

    dummy_tensor = torch.rand((10, 3)).to(device)

    output = model(dummy_tensor)

    logging.info(f"\t ➡️ Input shape : {dummy_tensor.shape}\n")
    logging.info(f"\t ➡️ Output shape : {output.shape}\n")

    assert output.shape == (10, 1), "❌ Error : the ouptut should be size of (10, 1)\n"
    logging.info("✅ MorpHoloNet is working well\n")


def test_holosover(cfg, device) : 
    logging.info("🛠️ Test HoloSolver\n")

    width, height  = 128, 128

    dummy_image = torch.rand((width, height)).to(device)

    logging.info(f"\t ➡️ Input shape : {dummy_image.shape}\n")

    model = HoloSolver(cfg["physical_params"], cfg["nerf_params"], dummy_image)
    model.to(device)
    logging.info(f"✅ Model called : {model}\n")

    logging.info(f"🛠️ Test of forward pass\n")
    model.train()
    loss_physics, loss_bc, total_loss = model(dummy_image, with_bc = True)
    logging.info("✅ Forward Pass successfull.\n")
    logging.info(f"loss_physics : {loss_physics}")
    logging.info(f"loss_bc : {loss_bc}")
    logging.info(f"total_loss : {total_loss}")
    logging.info("✅ Backward Pass successfull.\n")

    logging.info(f"🛠️ Test of generate_output\n")
    debug_dir = "./debug_test_output"
    
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)

    model.eval()
    model.generate_output(debug_dir)

    if os.path.exists(os.path.join(debug_dir, "obj")) and os.path.exists(os.path.join(debug_dir, "intensity")):
        logging.info(f"✅ Files generated successfully in {debug_dir}\n")
    else:
        logging.error("❌ Files not created.\n")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"= Using device {device}")

    cfg = {"nerf_params" : 
           {"gaussian_proj" : 64,
            "gaussian_scale": 10.0},

           "physical_params":{
            "step_z" : 1.0,
            "z_max": 10, 
            "waveLength": 0.3991, 
            "physicalLength": 0.5,
            "phase_shift": 0.5 
           }
    }
    test_morpholonet(cfg["nerf_params"], device)
    logging.info("\n\n")
    test_holosover(cfg, device)
