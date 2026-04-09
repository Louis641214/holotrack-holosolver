# coding: utf-8

# External imports
import torch
import logging
import sys
import os
import shutil

from .morpholonet import MorpHoloNet
from .holosolver import HoloSolver
from .hash_grid import Hash_Grid
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

    model = HoloSolver(cfg["physical_params"], cfg["nerf_params"], cfg["regularization"], dummy_image)
    model.to(device)
    logging.info(f"✅ Model called : {model}\n")
    torch.set_printoptions(threshold=20000)
    logging.info(f"🛠️ Test normalization of x, y and z\n")
    logging.info(model.xx.shape)
    logging.info(model.xx[-1])
    logging.info(model.yy[0])

    logging.info(f"🛠️ Test of forward pass\n")
    model.train()
    loss_physics, loss_bc, weighted_loss_sparsity, weighted_loss_tv, total_loss = model(dummy_image)
    logging.info("✅ Forward Pass successfull.\n")
    logging.info(f"loss_physics : {loss_physics}")
    logging.info(f"loss_bc : {loss_bc}")
    logging.info(f"weighted_loss_sparsity : {weighted_loss_sparsity}")
    logging.info(f"weighted_loss_tv : {weighted_loss_tv}")
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

    logging.info(f"🛠️ Test of reconstruct_hologram\n")
    image = model.reconstruct_hologram(dummy_image)

    assert image.shape == (width, height), "❌ Error : the reconstructed hologram should be size of (128, 128)\n"
    assert not torch.isnan(image).any(), "❌ Error: The reconstructed hologram contains NaN values!\n"

    logging.info("✅ reconstruct_hologram is working well\n")

    
def test_hash_grid(device) : 
    logging.info(" 🛠️ Test Hash Grid\n")
    model = Hash_Grid()
    model.to(device)
    logging.info(f"✅ Model called : {model}\n")

    dummy_tensor = torch.rand((10, 3)).to(device)

    output = model(dummy_tensor)

    logging.info(f"\t ➡️ Input shape : {dummy_tensor.shape}\n")
    logging.info(f"\t ➡️ Output shape : {output.shape}\n")
    
    assert output.shape == (10, 1), "❌ Error : the ouptut should be size of (10, 1)\n"
    logging.info("✅ Hash Grid is working well\n")




if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"= Using device {device}")
    
    cfg = {"nerf_params" : 
           {"model" : "MorpHoloNet",
           "gaussian_proj" : 64,
            "gaussian_scale": 10.0},

            "regularization" :{
                "with_bc": True},

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
    test_hash_grid(device)
    

    cfg = {"nerf_params" : {
            "model" : "Hash_Grid"},
            
            "regularization" :{
                "with_bc": True,
                "with_sparsity": True,
                "sparsity_weight": 1e-4,
                "with_tv": True,
                "tv_weight": 1e-5},

            "physical_params":{
                "step_z" : 1.0,
                "z_max": 10, 
                "waveLength": 0.3991, 
                "physicalLength": 0.5,
                "phase_shift": 0.5 
           }
    }

    test_holosover(cfg, device)
