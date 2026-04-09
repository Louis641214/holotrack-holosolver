# coding: utf-8

# Standard imports
import os

# External imports
import torch
import torch.nn
import shutil
import os


def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint(object):
    """
    Early stopping callback
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score, e=None):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False


def train(model, U_z0, optimizer, e, use_dtype, scaler):

    model.train()

    optimizer.zero_grad()
    
    with torch.autocast(device_type='cuda', dtype=use_dtype):
        loss_physics, loss_bc, weighted_loss_sparsity, weighted_loss_tv, total_loss, loss_pre_training, volume_3d = model(U_z0, e)
    
    scaler.scale(total_loss).backward()
    
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    scaler.step(optimizer)
    scaler.update()

    return loss_physics.item(), loss_bc.item(), weighted_loss_sparsity.item(), weighted_loss_tv.item(), total_loss.item(), loss_pre_training.item(), volume_3d, total_norm

def test(model, save_dir, e):
    model.eval()

    epoch_dir = os.path.join(save_dir, f"weights_{e}")
    os.makedirs(epoch_dir, exist_ok=True)

    if model.hash is True:
        model.generate_output_hash(epoch_dir)
    else : 
        model.generate_output(epoch_dir)
