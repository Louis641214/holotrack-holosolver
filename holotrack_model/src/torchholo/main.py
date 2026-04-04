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
import numpy as np
import random
import shutil

# Local imports
from . import data
from . import models
from . import optim
from . import utils

# Fixer la graine pour que chaque run soit identique
"""
seed = 302  # Tu peux tester 42, 0, ou 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
"""

def train(config):
    #----------Get CUDA device----------
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"= Using device {device}")

    #-------------Dataloading-------------
    logging.info("= Getting initial holographic image (U_z0)")
    data_config = config["data"]

    U_z0 = data.get_hologram(data_config)
    U_z0 = U_z0.to(device)

    #------------Target Blurring----------- 
    if data_config["blurring"]["activ"] is True : 
        target_blurring = data.TargetBlurring(data_config["blurring"])

    #------------Build the model-----------
    torch.set_float32_matmul_precision('high')
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, hologram = U_z0, device=device)
    model.to(device)
    model.Nerf = torch.compile(model.Nerf)

    #-----------Build the optimizer---------
    logging.info("= Optimizer")
    optim_config = config["optim"]
    
    nerf_params = []
    physics_params = []
    for name, param in model.named_parameters() : 
        if "Nerf" in name : 
            nerf_params.append(param)
        else : 
            physics_params.append(param)
    
    if optim_config["scheduler"]["warm_start"] is True:
        param_groups = [
        {"params": nerf_params}, 
        {"params": physics_params, "lr": 0.0}
        ]
    else :
        param_groups = [
        {"params": nerf_params}, 
        {"params": physics_params, "lr": optim_config["lr_physics"]}
        ]
    optimizer = optim.get_optimizer(optim_config, param_groups)
    
    #-----------Scheduler-------------
    if optim_config["scheduler"]["activ"] is True : 
        """
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=optim_config["scheduler"]["factor"],
            patience=optim_config["scheduler"]["patience"],
            threshold=optim_config["scheduler"]["threshold"],
            threshold_mode=optim_config["scheduler"]["threshold_mode"]
        )
        """
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=["scheduler"]["milestones"],
            gamma=optim_config["scheduler"]["gamma"])

    #-----------Stop criteria-------------
    if optim_config["stop_criteria"]["activ"] is True : 
        stop_criteria = optim.VolumeDelta(optim_config["stop_criteria"])

    # --------Build the callbacks-------
    logging_config = config["logging"]

    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    #----------summary script of the experiment----------
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

    #----------early stopping callback [NOT ACTIVE]--------------
    
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )
    
    progress_bar = tqdm.trange(config["nepochs"], desc="Training")
    
    #--------------Tensorboard target image-------------------
    target_image = U_z0.detach().cpu()
    tensorboard_writer.add_image("Visu/True_Holo", target_image.unsqueeze(0), 0)
    clean_target = U_z0.clone()

    loss_buffer = 0.0

    #-----------Create Result folder--------
    save_dir = config["test"]["save_dir"]

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    #-------------Train LOOP--------------
    for e in progress_bar:
        
        #-----------Blurring regularization---------------
        if data_config["blurring"]["activ"] is True :
            if e < target_blurring.blur_epochs :
                if e%500==0 :
                    U_z0, sigma = target_blurring.update(clean_target, e)
                    tensorboard_writer.add_scalar("Params/sigma", sigma, e)
            else :
                U_z0 = clean_target
                tensorboard_writer.add_scalar("Params/sigma", 0.0, e)
                data_config["blurring"]["activ"] = False
        
        #-----------BARF regularization----------------
        if model_config["nerf_params"]["barf"] is True :
            current_alpha = model.update_barf_progress(e, model_config["nerf_params"]["barf_epochs"])
            tensorboard_writer.add_scalar("Params/BARF_Progress", current_alpha, e)

        #------------TRAIN call--------------  

        loss_physics, loss_bc, weighted_loss_sparsity, weighted_loss_tv, total_loss, loss_pre_training, volume_3d, total_norm = utils.train(model, U_z0, optimizer, e)

        #-----------Parameters-Norm------------
        total_param_norm = 0.0
        for p in model.parameters():
            param_norm = p.detach().data.norm(2) # L2 norm du tenseur de poids
            total_param_norm += param_norm.item() ** 2
        total_param_norm = total_param_norm ** 0.5
        tensorboard_writer.add_scalar("Params/Weights_Norm", total_param_norm, e)

        #------------Scheduler call-----------
        if optim_config["scheduler"]["activ"] is True : 
            loss_buffer+=total_loss
            if e>2600 and e%200==0:
                avg_loss_200 = loss_buffer/200
                logging.info(f"avg loss : {avg_loss_200}")
                scheduler.step(avg_loss_200)
                loss_buffer = 0.0
                #optimizer.param_groups[1]["lr"] = optim_config["lr_physics"]
                tensorboard_writer.add_scalar("Params/Lr_Nerf", optimizer.param_groups[0]["lr"], e)
                tensorboard_writer.add_scalar("Params/Lr_Physics", optimizer.param_groups[1]["lr"], e)
        
        #------------TQDM loss info------------
        if e <= model_config["pre_training"]["epochs"] and model_config["pre_training"]["activ"] is True :
            progress_bar.set_description("Pre-training")
        else : 
            progress_bar.set_description("Training")
        progress_bar.set_postfix({"loss": f"{total_loss:.8f}"})

        #-----------TensorBoard adding-------------
        tensorboard_writer.add_scalar("Params/Norm_grad", total_norm, e)
        tensorboard_writer.add_scalar("Loss/total", total_loss, e)
        tensorboard_writer.add_scalar("Loss/Physics", loss_physics, e)
        tensorboard_writer.add_scalar("Loss/BC", loss_bc, e)
        tensorboard_writer.add_scalar("Loss/Sparcity", weighted_loss_sparsity, e)
        tensorboard_writer.add_scalar("Loss/TV", weighted_loss_tv, e)
        tensorboard_writer.add_scalar("Loss/pre_training", loss_pre_training, e)
        tensorboard_writer.add_scalar("Params/Phase_shift", model.phase_shift.item(), e)
        tensorboard_writer.add_scalar("Params/Incident_Light", model.incident_light.item(), e)
        
        #-----------Reconstruct hologram-------------
        if e%200==0 : 
            if model.hash is True: 
                raw_holo = model.reconstruct_hologram_hash().detach().cpu()
            else : 
                raw_holo = model.reconstruct_hologram().detach().cpu()
            norm_holo = ((raw_holo - raw_holo.min()) / (raw_holo.max() - raw_holo.min() +1e-8))
            tensorboard_writer.add_image("Visu/Rec_Holo", norm_holo.unsqueeze(0), e)
            
            #----------Reconstruct hologram with blurring regulariuzation----------
            if data_config["blurring"]["activ"] :
                tensorboard_writer.add_image("Visu/Actual_Holo", U_z0.detach().cpu().unsqueeze(0), e)
        
        #----------Stop Criteria update----------
        if optim_config["stop_criteria"]["activ"] is True : 
            if e % stop_criteria.check_interval == 0 :
                done, relative_change = stop_criteria.update(volume_3d)
                tensorboard_writer.add_scalar("Params/Stop_criteria", relative_change, e)
                logging.info(f"Volume fixed for {stop_criteria.current_patience*stop_criteria.check_interval}")
                logging.info(f"Patience : {stop_criteria.current_patience}")
                if done :
                    logging.info(f"STOP TRAINING FORCED")
                    model_checkpoint.update(total_loss)
                    logging.info("=Generation of volume")
                    test_config = config["test"]
                    utils.test(model, test_config)
                    break

        #----------Save Model weights----------
        if (e>300 and e%1000==0) or (e==model_config["pre_training"]["epochs"]):
            model_checkpoint.update(total_loss)
            logging.info("=Generation of volume")
            test_config = config["test"]
            utils.test(model, save_dir, e)

    #---------End training resume----------
    phase_shift, incident_light = model.get_internal_values()
    logging.info("-" * 30)
    logging.info("END TRAINING")
    logging.info(f"Stopped at epoch    : {e}")
    logging.info(f"MODEL SAVED TO      : {str(logdir / 'best_model.pt')}")
    logging.info(f"Total   Loss        : {total_loss}")
    logging.info(f"Physics Loss        : {loss_physics}")
    logging.info(f"BC      Loss        : {loss_bc}")
    logging.info(f"Final Phase Shift   : {phase_shift:.6f}")
    logging.info(f"Final Incident Light: {incident_light:.6f}")
    logging.info("-" * 30)


def test(config):

    #----------Get CUDA device----------
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"= Using device {device}")

    #-------------Dataloading-------------
    logging.info("= Getting initial holographic image (U_z0)")
    torch.set_float32_matmul_precision('high')
    data_config = config["data"]

    U_z0 = data.get_hologram(data_config)
    U_z0 = U_z0.to(device)
    
    #------------Build the model-----------
    logging.info("= Initializig Model")
    model_config = config["model"]
    model = models.build_model(model_config, hologram = U_z0, device=device)
    model.to(device)
    model.Nerf = torch.compile(model.Nerf)

    #----------Get weights---------------
    test_config = config["test"]
    weights_path = test_config["weights_path"]

    if weights_path is None or not os.path.isfile(weights_path) : 
        logging.error(f"Weights not found or not specified at {weights_path}")
        sys.exit(-1)

    logging.info(f"= Loading Weights from {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)

    #---------Start Test----------------
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
