# coding: utf-8
import torch
import torch.nn as nn
import numpy as np

class PositionalEncodingBarf(nn.Module):

    def __init__(self, gaussian_projection: int, gaussian_scale: float = 1.0, **kwargs):
        super(PositionalEncodingBarf, self).__init__()

        self._kernel_dtype = kwargs.get('dtype', None)

        self.gauss_proj = int(gaussian_projection)
        self.gauss_scale= float(gaussian_scale)

        self.register_buffer("alpha_progress", torch.tensor(1.0))

        if self.gauss_proj <= 0:
            self.proj = nn.Linear(in_features=2, out_features=2, bias=False, dtype=self._kernel_dtype)
            self.z_proj = nn.Linear(in_features=1, out_features=1, bias=False, dtype=self._kernel_dtype)
        else : 
            self.proj = nn.Linear(in_features=2, out_features=self.gauss_proj, bias=False, dtype=self._kernel_dtype)
            self.z_proj = nn.Linear(in_features=1, out_features=self.gauss_proj, bias=False, dtype=self._kernel_dtype)
        
        self.init_weights()
        
    def init_weights(self) : 

        for param in self.parameters() : 
            param.requires_grad = False
       
        if self.gauss_proj<=0 : 
            nn.init.eye_(self.proj.weight)
            nn.init.eye_(self.z_proj.weight)
        else : 
            nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=self.gauss_scale)
            nn.init.trunc_normal_(self.z_proj.weight, mean=0.0, std=self.gauss_scale)

            with torch.no_grad():
                # Trie XY
                magnitudes_xy = torch.norm(self.proj.weight, dim=1)
                indices_xy = torch.argsort(magnitudes_xy)
                self.proj.weight.copy_(self.proj.weight[indices_xy])
                
                # Trie Z
                magnitudes_z = torch.norm(self.z_proj.weight, dim=1)
                indices_z = torch.argsort(magnitudes_z)
                self.z_proj.weight.copy_(self.z_proj.weight[indices_z])

    def _get_barf_mask(self):
        """Calcule le masque d'activation progressif des fréquences."""
        if self.gauss_proj <= 0:
            return 1.0
            
        L = self.gauss_proj
        # alpha glisse de 0 à L
        start_alpha = 6.0
        alpha = start_alpha + self.alpha_progress * (L - start_alpha)
        
        k = torch.arange(L, device=self.alpha_progress.device, dtype=self.proj.weight.dtype)
        weight = (alpha - k).clamp(min=0.0, max=1.0)
        mask = (1.0 - torch.cos(np.pi * weight)) / 2.0
        
        return mask # Shape: (L,)

    def forward(self, inputs):

        xy_inputs = inputs[..., :2]
        z_inputs = inputs[..., 2:]

        x_proj = 2.0 * np.pi * xy_inputs
        x_proj = self.proj(x_proj)

        x_proj_sin = torch.sin(x_proj)
        x_proj_cos = torch.cos(x_proj)

        mask = self._get_barf_mask()
        x_proj_sin = x_proj_sin * mask
        x_proj_cos = x_proj_cos * mask

        xy_output = torch.cat([x_proj_sin, x_proj_cos], dim=-1)

        z_output = self.z_positional_encoding(z_inputs)

        output = torch.cat([xy_output, z_output], dim=-1)
        return output

    def z_positional_encoding(self, z):
        z_proj = 2.0 * np.pi * z
        z_proj = self.z_proj(z_proj)

        z_proj_sin = torch.sin(z_proj)
        z_proj_cos = torch.cos(z_proj)

        mask = self._get_barf_mask()
        z_proj_sin = z_proj_sin * mask
        z_proj_cos = z_proj_cos * mask

        pos_encoding = torch.cat([z_proj_sin, z_proj_cos], dim=-1)
        return pos_encoding