import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image

from .morpholonet import MorpHoloNet
from .physics_model import angular_spectrum_propagator

class HoloSolver(nn.Module) :
    def __init__(self, physical_params, nerf_params, U_z0):
        super(HoloSolver, self).__init__()

        self.Nerf = MorpHoloNet(nerf_params)
        
        # ==============================================================================
        # PARAMETERS 1 : Physics parameters
        # ============================================================================== 
        self.height, self.width = U_z0.shape
        self.waveLength = physical_params["waveLength"]
        self.physicalLength = physical_params["physicalLength"]
        self.z_max = physical_params["z_max"]
        self.dz = physical_params["step_z"]
        self.z_min = 0 + self.dz
        self.segment_size = self.width
        self.U_incident_avg_real = torch.sqrt(torch.mean(U_z0)).item()
        # ==============================================================================
        # PARAMETERS 2 : Lernable parameters
        # ==============================================================================
        self.phase_shift = nn.Parameter(torch.tensor(physical_params["phase_shift"], dtype=torch.float32))
        self.incident_light = nn.Parameter(torch.tensor(self.U_incident_avg_real, dtype=torch.float32))
        
        # ==============================================================================
        # OPTIMISATION : init BUFFERS for forward_physics and forward_BC
        # ==============================================================================
        self._init_physics_buffer()
        self._init_bc_buffer()
  
        
    def _init_physics_buffer(self) : 
        x_range = torch.arange(1, self.width+1, dtype=torch.float32)/self.width
        y_range = torch.arange(1, self.height+1, dtype=torch.float32)/self.height
        xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
        self.register_buffer('xx', xx)
        self.register_buffer('yy', yy)

        z_values = torch.arange(self.z_min, self.z_max + (self.dz/100), step=self.dz, dtype=torch.float32).flip(-1)
        self.register_buffer('z_values', z_values)

        ones_grid = torch.ones((self.width, self.height), dtype=torch.float32)
        zeros_grid = torch.zeros((self.width, self.height), dtype=torch.float32)
        self.register_buffer('ones_grid', ones_grid)
        self.register_buffer('zeros_grid', zeros_grid)

        scalar_zero = torch.tensor(0.0, dtype=torch.float32)
        scalar_one = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer('scalar_zero', scalar_zero)
        self.register_buffer('scalar_one', scalar_one)

    def _init_bc_buffer(self) :
        offset = 0.0

        x_BC = torch.arange(1, self.width + 1, dtype=torch.float32)/self.width
        y_BC = torch.arange(1, self.height + 1, dtype=torch.float32)/self.height
        z_BC = torch.arange(0, self.z_max + (self.dz/100), step=self.dz, dtype=torch.float32)/self.z_max

        xx_BC, zz_BC = torch.meshgrid(x_BC, z_BC, indexing='ij')
        yy_0 = torch.full_like(xx_BC, 0.0 + (offset / self.height))
        yy_1 = torch.full_like(xx_BC, 1.0 - (offset / self.height))

        self.register_buffer('tensor_array_BC_0', torch.stack([xx_BC, yy_0, zz_BC], dim=-1).reshape(-1, 3))
        self.register_buffer('tensor_array_BC_1', torch.stack([xx_BC, yy_1, zz_BC], dim=-1).reshape(-1, 3))

        yy_BC, zz_BC = torch.meshgrid(y_BC, z_BC, indexing='ij')
        xx_0 = torch.full_like(yy_BC, 0.0 + (offset / self.width))
        xx_1 = torch.full_like(yy_BC, 1.0 - (offset / self.width))

        self.register_buffer('tensor_array_BC_2',torch.stack([xx_0, yy_BC, zz_BC], dim=-1).reshape(-1, 3))
        self.register_buffer('tensor_array_BC_3',torch.stack([xx_1, yy_BC, zz_BC], dim=-1).reshape(-1, 3))
    
    def forward_physics(self, U_z0):
        device = self.scalar_zero.device

        loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        complex_i = torch.complex(self.scalar_zero, self.scalar_one)

        U_z_following_prop = torch.complex(self.zeros_grid, self.zeros_grid)

        for z in self.z_values:
            if torch.isclose(z, torch.tensor(self.z_max, dtype=torch.float32, device=device)):
                z_following = torch.full_like(self.xx, z / self.z_max)
                z_preceding = torch.full_like(self.xx, (z - self.dz) / self.z_max)

                tensor_array_following = torch.stack([self.xx, self.yy, z_following], dim=-1).reshape(-1, 3)
                tensor_array_preceding = torch.stack([self.xx, self.yy, z_preceding], dim=-1).reshape(-1, 3)

                object_following = self.Nerf(tensor_array_following).reshape(self.width, self.height)
                object_preceding = self.Nerf(tensor_array_preceding).reshape(self.width, self.height)

                real_following_ref = self.ones_grid * self.incident_light
                U_z_following_ref = torch.complex(real_following_ref, self.zeros_grid)
                
                phase_shift_complex = torch.complex(self.phase_shift, self.scalar_zero)
                object_preceding_complex = torch.complex(object_preceding, self.zeros_grid)
                
                phase_delay = torch.exp(complex_i * phase_shift_complex * object_preceding_complex)
                U_z_following_ref *= phase_delay

                U_z_following_prop = angular_spectrum_propagator(image=U_z_following_ref, depth=self.dz, 
                                                                 device=device, segment_size=self.segment_size,
                                                                 physicalLength = self.physicalLength,
                                                                 waveLength=self.waveLength)

                loss += 0.5*torch.mean(torch.square(object_following))

            elif torch.isclose(z, torch.tensor(self.z_min, dtype=torch.float32, device=device)):
                z_preceding = torch.full_like(self.xx, (z - self.dz) / self.z_max)
                tensor_array_preceding = torch.stack([self.xx, self.yy, z_preceding], dim=-1).reshape(-1, 3)

                object_preceding = self.Nerf(tensor_array_preceding).reshape(self.width, self.height)

                phase_shift_complex = torch.complex(self.phase_shift, self.scalar_zero)
                object_preceding_complex = torch.complex(object_preceding, self.zeros_grid)
                phase_delay = torch.exp(complex_i * phase_shift_complex * object_preceding_complex)
                U_z_following_prop *= phase_delay

                U_z_following_prop = angular_spectrum_propagator(image=U_z_following_prop, depth=self.dz, 
                                                                 device=device, segment_size=self.segment_size,
                                                                 physicalLength = self.physicalLength,
                                                                 waveLength=self.waveLength)
                
                U_z_following_prop_intensity = torch.square(torch.abs(U_z_following_prop))

                loss += torch.mean(torch.square(U_z0 - U_z_following_prop_intensity))
                loss += 0.5*torch.mean(torch.square(object_preceding))

            else:
                z_preceding = torch.full_like(self.xx, (z - self.dz) / self.z_max)
                tensor_array_preceding = torch.stack([self.xx, self.yy, z_preceding], dim=-1).reshape(-1, 3)

                object_preceding = self.Nerf(tensor_array_preceding).reshape(self.width, self.height)

                phase_shift_complex = torch.complex(self.phase_shift, self.scalar_zero)
                object_preceding_complex = torch.complex(object_preceding, self.zeros_grid)
                phase_delay = torch.exp(complex_i * phase_shift_complex * object_preceding_complex)
                U_z_following_prop *= phase_delay

                U_z_following_prop = angular_spectrum_propagator(image=U_z_following_prop, depth=self.dz, 
                                                                 device=device, segment_size=self.segment_size,
                                                                 physicalLength = self.physicalLength,
                                                                 waveLength=self.waveLength)
        return loss

    def forward_BC(self) : 
        device = self.scalar_zero.device
        loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        object_BC_0 = self.Nerf(self.tensor_array_BC_0)
        object_BC_1 = self.Nerf(self.tensor_array_BC_1)

        loss += 0.5*torch.mean(torch.square(object_BC_0))
        loss += 0.5*torch.mean(torch.square(object_BC_1))

        object_BC_2 = self.Nerf(self.tensor_array_BC_2)
        object_BC_3 = self.Nerf(self.tensor_array_BC_3)

        loss += 0.5*torch.mean(torch.square(object_BC_2))
        loss += 0.5*torch.mean(torch.square(object_BC_3))

        return loss
    
    def get_internal_values(self):
        return self.phase_shift.item(), self.incident_light.item()

    def forward(self, U_z0, with_bc):
        loss_bc = torch.tensor(0.0, dtype=torch.float32, device=self.scalar_zero.device)
        loss_physics = self.forward_physics(U_z0)
        if with_bc :
            loss_bc = self.forward_BC()
        total_loss = loss_physics + loss_bc
        return loss_physics, loss_bc, total_loss
    
    @torch.no_grad()
    def generate_output(self, save_dir) : 
        obj_dir = os.path.join(save_dir, 'obj')
        intensity_dir = os.path.join(save_dir, 'intensity')
        os.makedirs(obj_dir, exist_ok=True)
        os.makedirs(intensity_dir, exist_ok=True)

        device = self.scalar_zero.device
        print(f"Generating results on {device}...")

        # ======================================================================
        # PARTIE 1 : Reconstruct 3D Object (NeRF Query)
        # ======================================================================
        print("Reconstructing 3D Object slices...")
        volume_shape = (self.width, self.height, len(self.z_values))
        obj_volume = np.zeros(volume_shape, dtype=np.float32)

        for i, z in enumerate(self.z_values):
            z_norm = z / self.z_max
            z_filled = torch.full_like(self.xx, z_norm)

            tensor_array = torch.stack([self.xx, self.yy, z_filled], dim=-1).reshape(-1, 3)
            obj_slice = self.Nerf(tensor_array).reshape(self.width, self.height)
            obj_npy = obj_slice.cpu().numpy()

            obj_volume[:, :, i] = obj_npy

            img = Image.fromarray(obj_npy)
            img.save(os.path.join(obj_dir, f"obj_{z.item():.1f}.tif"))
        
        np.save(os.path.join(obj_dir, "volume_3d.npy"), obj_volume)

        # ======================================================================
        # PARTIE 2 : Propagation Physique (Backward Propagation)
        # ======================================================================
        print("Simulating Light Propagation...")
        complex_i = torch.complex(self.scalar_zero, self.scalar_one)
        U_z_following_prop = torch.complex(self.zeros_grid, self.zeros_grid)
        real_following_ref = self.ones_grid * self.incident_light
        U_z_following_ref = torch.complex(real_following_ref, self.zeros_grid)
        phase_shift_complex = torch.complex(self.phase_shift, self.scalar_zero)
        
        for z in self.z_values:
            if torch.isclose(z, torch.tensor(self.z_max, dtype=torch.float32, device=device)):
                
                U_z_following_ref_intensity = torch.square(torch.abs(U_z_following_ref)).cpu().numpy()
                U_z_following_ref_intensity = Image.fromarray(U_z_following_ref_intensity)
                U_z_following_ref_intensity.save(os.path.join(intensity_dir, f"Intensity_MorpHoloNet_{z.item():.1f}.tif"))
            
            elif torch.isclose(z, torch.tensor(self.z_max - self.dz, dtype=torch.float32, device=device)):
                z_following = torch.full_like(self.xx, z / self.z_max)
    
                tensor_array_following = torch.stack([self.xx, self.yy, z_following], dim=-1).reshape(-1, 3)
                
                object_following = self.Nerf(tensor_array_following).reshape(self.width, self.height)
                
                object_following_complex = torch.complex(object_following, self.zeros_grid)
                
                phase_delay = torch.exp(complex_i * phase_shift_complex * object_following_complex)
                U_z_following_prop = U_z_following_ref * phase_delay


                U_z_following_prop = angular_spectrum_propagator(image=U_z_following_prop, depth=self.dz, 
                                                                 device=device, segment_size=self.segment_size,
                                                                 physicalLength = self.physicalLength,
                                                                 waveLength=self.waveLength)
                U_z_following_prop_intensity = torch.square(torch.abs(U_z_following_prop)).cpu().numpy()
                U_z_following_prop_intensity = Image.fromarray(U_z_following_prop_intensity)
                U_z_following_prop_intensity.save(os.path.join(intensity_dir, f"Intensity_MorpHoloNet_{z.item():.1f}.tif"))


            elif torch.isclose(z, torch.tensor(self.z_min, dtype=torch.float32, device=device)):
                z_following = torch.full_like(self.xx, z / self.z_max)
    
                tensor_array_following = torch.stack([self.xx, self.yy, z_following], dim=-1).reshape(-1, 3)
                
                object_following = self.Nerf(tensor_array_following).reshape(self.width, self.height)
                
                
                object_following_complex = torch.complex(object_following, self.zeros_grid)
                
                phase_delay = torch.exp(complex_i * phase_shift_complex * object_following_complex)
                U_z_following_prop *= phase_delay


                U_z_following_prop = angular_spectrum_propagator(image=U_z_following_prop, depth=self.dz, 
                                                                 device=device, segment_size=self.segment_size,
                                                                 physicalLength = self.physicalLength,
                                                                 waveLength=self.waveLength)
                U_z_following_prop_intensity = torch.square(torch.abs(U_z_following_prop)).cpu().numpy()
                U_z_following_prop_intensity = Image.fromarray(U_z_following_prop_intensity)
                U_z_following_prop_intensity.save(os.path.join(intensity_dir, f"Intensity_MorpHoloNet_{z.item():.1f}.tif"))

                U_z_following_prop = angular_spectrum_propagator(image=U_z_following_prop, depth=self.dz, 
                                                                 device=device, segment_size=self.segment_size,
                                                                 physicalLength = self.physicalLength,
                                                                 waveLength=self.waveLength)
                U_z_following_prop_intensity = torch.square(torch.abs(U_z_following_prop)).cpu().numpy()
                U_z_following_prop_intensity = Image.fromarray(U_z_following_prop_intensity)
                U_z_following_prop_intensity.save(os.path.join(intensity_dir, f"Intensity_MorpHoloNet_0.tif"))

            else:
                z_following = torch.full_like(self.xx, z / self.z_max)
    
                tensor_array_following = torch.stack([self.xx, self.yy, z_following], dim=-1).reshape(-1, 3)
                
                object_following = self.Nerf(tensor_array_following).reshape(self.width, self.height)

                object_following_complex = torch.complex(object_following, self.zeros_grid)
                
                phase_delay = torch.exp(complex_i * phase_shift_complex * object_following_complex)
                U_z_following_prop *= phase_delay


                U_z_following_prop = angular_spectrum_propagator(image=U_z_following_prop, depth=self.dz, 
                                                                 device=device, segment_size=self.segment_size,
                                                                 physicalLength = self.physicalLength,
                                                                 waveLength=self.waveLength)
                U_z_following_prop_intensity = torch.square(torch.abs(U_z_following_prop)).cpu().numpy()
                U_z_following_prop_intensity = Image.fromarray(U_z_following_prop_intensity)
                U_z_following_prop_intensity.save(os.path.join(intensity_dir, f"Intensity_MorpHoloNet_{z.item():.1f}.tif"))

        print("Generation done")

    @torch.no_grad()
    def reconstruct_hologram(self, U_z0) :
        device = U_z0.device

        complex_i = torch.complex(self.scalar_zero, self.scalar_one)
        U_z_following_prop = torch.complex(self.zeros_grid, self.zeros_grid)
        real_following_ref = self.ones_grid * self.incident_light
        U_z_following_ref = torch.complex(real_following_ref, self.zeros_grid)
        phase_shift_complex = torch.complex(self.phase_shift, self.scalar_zero)
        
        for z in self.z_values:
            if torch.isclose(z, torch.tensor(self.z_max, dtype=torch.float32, device=device)):
                continue
            
            elif torch.isclose(z, torch.tensor(self.z_max - self.dz, dtype=torch.float32, device=device)):
                z_following = torch.full_like(self.xx, z / self.z_max)
    
                tensor_array_following = torch.stack([self.xx, self.yy, z_following], dim=-1).reshape(-1, 3)
                
                object_following = self.Nerf(tensor_array_following).reshape(self.width, self.height)
                
                object_following_complex = torch.complex(object_following, self.zeros_grid)
                
                phase_delay = torch.exp(complex_i * phase_shift_complex * object_following_complex)
                U_z_following_prop = U_z_following_ref * phase_delay


                U_z_following_prop = angular_spectrum_propagator(image=U_z_following_prop, depth=self.dz, 
                                                                device=device, segment_size=self.segment_size,
                                                                physicalLength = self.physicalLength,
                                                                waveLength=self.waveLength)


            elif torch.isclose(z, torch.tensor(self.z_min, dtype=torch.float32, device=device)):
                z_following = torch.full_like(self.xx, z / self.z_max)
    
                tensor_array_following = torch.stack([self.xx, self.yy, z_following], dim=-1).reshape(-1, 3)
                
                object_following = self.Nerf(tensor_array_following).reshape(self.width, self.height)
                
                
                object_following_complex = torch.complex(object_following, self.zeros_grid)
                
                phase_delay = torch.exp(complex_i * phase_shift_complex * object_following_complex)
                U_z_following_prop *= phase_delay


                U_z_following_prop = angular_spectrum_propagator(image=U_z_following_prop, depth=self.dz, 
                                                                device=device, segment_size=self.segment_size,
                                                                physicalLength = self.physicalLength,
                                                                waveLength=self.waveLength)


                U_0_following_prop = angular_spectrum_propagator(image=U_z_following_prop, depth=self.dz, 
                                                                device=device, segment_size=self.segment_size,
                                                                physicalLength = self.physicalLength,
                                                                waveLength=self.waveLength)
                

            else:
                z_following = torch.full_like(self.xx, z / self.z_max)
    
                tensor_array_following = torch.stack([self.xx, self.yy, z_following], dim=-1).reshape(-1, 3)
                
                object_following = self.Nerf(tensor_array_following).reshape(self.width, self.height)

                object_following_complex = torch.complex(object_following, self.zeros_grid)
                
                phase_delay = torch.exp(complex_i * phase_shift_complex * object_following_complex)
                U_z_following_prop *= phase_delay


                U_z_following_prop = angular_spectrum_propagator(image=U_z_following_prop, depth=self.dz, 
                                                                device=device, segment_size=self.segment_size,
                                                                physicalLength = self.physicalLength,
                                                                waveLength=self.waveLength)

        
        return torch.square(torch.abs(U_0_following_prop))
