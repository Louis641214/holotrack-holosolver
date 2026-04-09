'''
Original publication: Tancik, M. et al. Fourier features let networks learn high frequency functions in low dimensional domains. Adv. Neural inf. Process. Syst. 33, 7537-7547 (2020).
Original code: https://github.com/titu1994/tf_fourier_features/blob/master/tf_fourier_features/fourier_features.py
Original license:

MIT License

Copyright (c) 2020 Somshubra Majumdar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modified by Jihwan Kim
'''

import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, gaussian_projection: int, gaussian_scale: float = 1.0, **kwargs):
        super(PositionalEncoding, self).__init__()

        self._kernel_dtype = kwargs.get('dtype', None)

        self.gauss_proj = int(gaussian_projection)
        self.gauss_scale= float(gaussian_scale)

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

    def forward(self, inputs):

        xy_inputs = inputs[..., :2]
        z_inputs = inputs[..., 2:]

        x_proj = 2.0 * np.pi * xy_inputs
        x_proj = self.proj(x_proj)

        x_proj_sin = torch.sin(x_proj)
        x_proj_cos = torch.cos(x_proj)

        xy_output = torch.cat([x_proj_sin, x_proj_cos], dim=-1) #On cat les channels [B, 64+64 = 128]

        z_output = self.z_positional_encoding(z_inputs)
        #z_output = torch.reshape(z_output, [-1, self.gauss_proj * 2])

        output = torch.cat([xy_output, z_output], dim=-1) # On cat les channels [B, 128 + 128 = 256]
        return output

    def z_positional_encoding(self, z):
        z_proj = 2.0 * np.pi * z
        z_proj = self.z_proj(z_proj)

        z_proj_sin = torch.sin(z_proj)
        z_proj_cos = torch.cos(z_proj)

        pos_encoding = torch.cat([z_proj_sin, z_proj_cos], dim=-1) # On cat les channels [B, 64+64 = 128]
        return pos_encoding
