# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib

# External imports
import yaml
import torch
#import torchinfo.torchinfo as torchinfo
import tqdm
from torch.utils.tensorboard import SummaryWriter

# Local imports
from . import data
from . import models
from . import optim
from . import utils


def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"= Using device {device}")

    # Dataloading
    logging.info("= Getting initial holographic image (U_z0)")
    data_config = config["data"]

    U_z0 = data.get_hologram(data_config)
    U_z0 = U_z0.to(device)

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, hologram = U_z0)
    model.to(device)

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
    input_size = U_z0.shape
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + "## Summary of the model architecture\n"
        + str(model) + "\n\n"
    )
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)

    tensorboard_writer = SummaryWriter(log_dir=logdir)
    tensorboard_writer.add_text("Experiment summary", summary_text)

    # Define the early stopping callback
    '''
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )
    '''
    progress_bar = tqdm.trange(config["nepochs"], desc="Training")
    
    #Tensorboard target image 
    target_image = U_z0.detach().cpu()
    tensorboard_writer.add_image("Visu/True_Holo", target_image.unsqueeze(0), 0)
    
    for e in progress_bar:
        # Train 1 epoch
        loss_physics, loss_bc, total_loss = utils.train(model, U_z0, optimizer, with_bc=model_config["with_bc"])

        '''
        progress_bar.set_postfix({"loss": f"{total_loss:.4f}", 
                                  "best": "[>> BETTER <<]" if updated else "No"})
        '''
        progress_bar.set_postfix({"loss": f"{total_loss:.8f}"})

        tensorboard_writer.add_scalar("Loss/total", total_loss, e)
        tensorboard_writer.add_scalar("Loss/Physics", loss_physics, e)
        tensorboard_writer.add_scalar("Loss/BC", loss_bc, e)
        tensorboard_writer.add_scalar("Params/Phase_shift", model.phase_shift.item(), e)
        tensorboard_writer.add_scalar("Params/Incident_Light", model.incident_light.item(), e)

        if e%50==0 : 
            raw_holo = model.reconstruct_hologram(U_z0).detach().cpu()
            norm_holo = ((raw_holo - raw_holo.min()) / (raw_holo.max() - raw_holo.min() +1e-8))
            tensorboard_writer.add_image("Visu/Rec_Holo", norm_holo.unsqueeze(0), e)


    phase_shift, incident_light = model.get_internal_values()
    torch.save(model.state_dict(), str(logdir / "best_model.pt"))
    logging.info("-" * 30)
    logging.info("END TRAINING")
    logging.info(f"MODEL SAVED TO      : {str(logdir / 'best_model.pt')}")
    logging.info(f"Total   Loss        : {total_loss}")
    logging.info(f"Physics Loss        : {loss_physics}")
    logging.info(f"BC      Loss        : {loss_bc}")
    logging.info(f"Final Phase Shift   : {phase_shift:.6f}")
    logging.info(f"Final Incident Light: {incident_light:.6f}")
    logging.info("-" * 30)

# NOTE : pas encore implémenté
def test(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"= Using device {device}")

    logging.info("= Getting initial holographic image (U_z0)")
    data_config = config["data"]

    U_z0 = data.get_hologram(data_config)
    U_z0 = U_z0.to(device)

    logging.info("= Initializig Model")
    model_config = config["model"]
    model = models.build_model(model_config, hologram = U_z0)
    model.to(device)

    test_config = config["test"]
    weights_path = test_config["weights_path"]

    if weights_path is None or not os.path.isfile(weights_path) : 
        logging.error(f"Weights not found or not specified at {weights_path}")
        sys.exit(-1)

    logging.info(f"= Loading Weights from {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)

    logging.info("= Data Generation...")
    utils.test(model, test_config)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")
