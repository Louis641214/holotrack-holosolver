# coding: utf-8

# External imports
import torch
import torch.nn as nn
import logging

def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim

class VolumeDelta:
    def __init__(self, cfg):
        self.max_patience = cfg["patience"]
        self.check_interval = cfg["interval"]
        self.delta = cfg["delta"]

        self.current_patience = 0
        self.prev_volume = None 

    def update(self, current_volume):
        
        done = False
        relative_change = 0.0
        if self.prev_volume is not None :
            delta = torch.mean(torch.abs(current_volume - self.prev_volume)).item()
            mean_mass = torch.mean(torch.abs(current_volume)).item() + 1e-8
            relative_change = delta/mean_mass

            if relative_change < self.delta:
                self.current_patience+=1
            else :
                self.current_patience=0

            if self.current_patience >=3 :
                done = True

        self.prev_volume = current_volume.detach().clone()
        
        return done, relative_change

class PingPongScheduler:
    def __init__(self, cfg, optimizer):
        self.lr_dropped = False
        self.grad_norm_limit = cfg["grad_norm_limit"]
        self.ping_pong_counter = 0
        self.ping_pong_limit = cfg["ping_pong_limit"]
        self.optimizer_params = optimizer.param_groups

    def step(self, grad_norm):
        if not self.lr_dropped and grad_norm > self.grad_norm_limit :
            self.ping_pong_counter+=1

            if self.ping_pong_counter > self.ping_pong_limit:
                for param_group in self.optimizer_params :
                    param_group["lr"] *=0.1
                self.lr_dropped = True
                logging.info("!!! LR DROPPED !!!")
            
        elif not self.lr_dropped:
            self.ping_pong_counter = max(0, self.ping_pong_counter - 1)
                
