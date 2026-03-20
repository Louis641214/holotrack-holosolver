import torch
import torch.nn as nn
import numpy as np
import os
import shutil
from PIL import Image
from torch.amp import autocast

from .morpholonet import MorpHoloNet
from .physics_model import PhysicsModel
from .hash_grid import Hash_Grid
from .holotrack import HoloTrack

class HoloSolver(nn.Module) :
    def __init__(self, physical_params, nerf_params, regularization_params, U_z0, device):
        super(HoloSolver, self).__init__()

        # ==============================================================================
        # MODEL : Select the right model to use
        # ============================================================================== 
        self.L2=False
        if nerf_params["model"] == "Hash_Grid" : 
            self.Nerf = Hash_Grid()
            self.hash = True
        elif nerf_params["model"] == "MorpHoloNet" : 
            self.Nerf = MorpHoloNet(nerf_params)
            self.hash = False
        elif nerf_params["model"] == "HoloTrack" : 
            self.Nerf = HoloTrack(nerf_params)
            self.hash = False

        self.device = device
        
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

        self.Physics_model = PhysicsModel(self.segment_size, self.physicalLength, self.dz, device)
        # ==============================================================================
        # PARAMETERS 2 : Lernable parameters
        # ==============================================================================
        self.phase_shift = nn.Parameter(torch.tensor(physical_params["phase_shift"], dtype=torch.float32))
        self.incident_light = nn.Parameter(torch.tensor(self.U_incident_avg_real, dtype=torch.float32))
        #self.register_buffer("phase_shift", torch.tensor(physical_params["phase_shift"], dtype=torch.float32))
        #elf.register_buffer("incident_light", torch.tensor(self.U_incident_avg_real, dtype=torch.float32))
        
        # ==============================================================================
        # OPTIMISATION : init BUFFERS for forward_physics and forward_BC
        # ==============================================================================
        if self.hash : 
            self._init_physics_buffer_hash()
            self._init_bc_buffer_hash()
        else : 
            self._init_physics_buffer()
            self._init_bc_buffer()

        # ==============================================================================
        # REGULARIZATION : Parameters of regularization
        # ==============================================================================
        self.with_sparsity = regularization_params.get("with_sparsity", False)
        self.sparsity_weight = regularization_params.get("sparsity_weight", 0.0)
        self.with_tv = regularization_params.get("with_tv", False)
        self.tv_weight = regularization_params.get("tv_weight", 0.0)
        self.with_bc = regularization_params.get("with_bc", False)

    def _init_physics_buffer(self) :
        '''
        Init all elements that will be used by forward_physics in order
        to optimize time during training"
        ''' 
        x_range = torch.arange(1, self.width+1, dtype=torch.float32)/self.width
        y_range = torch.arange(1, self.height+1, dtype=torch.float32)/self.height
        
        z_values = torch.arange(self.z_min, self.z_max + (self.dz/100), step=self.dz, dtype=torch.float32).flip(-1)
        self.register_buffer('z_values', z_values)

        z_eval = torch.cat([z_values, torch.tensor([self.z_min - self.dz], dtype=torch.float32)])
        
        z_eval_norm = z_eval / self.z_max

        Z, X, Y = torch.meshgrid(z_eval_norm, x_range, y_range, indexing='ij')

        coords_3d = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
        self.register_buffer("coords_3d", coords_3d)

        ones_grid = torch.ones((self.width, self.height), dtype=torch.float32)
        zeros_grid = torch.zeros((self.width, self.height), dtype=torch.float32)
        self.register_buffer('ones_grid', ones_grid)
        self.register_buffer('zeros_grid', zeros_grid)

        scalar_zero = torch.tensor(0.0, dtype=torch.float32)
        scalar_one = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer('scalar_zero', scalar_zero)
        self.register_buffer('scalar_one', scalar_one)
    
    def _init_bc_buffer(self) :
        """
        Init all elements that will be used by Forward_BC in order
        to optimize time during training
        """
        offset = 0.0
        x_BC = torch.arange(1, self.width + 1, dtype=torch.float32)/self.width
        y_BC = torch.arange(1, self.height + 1, dtype=torch.float32)/self.height
        z_BC = torch.arange(0, self.z_max + (self.dz/100), step=self.dz, dtype=torch.float32)/self.z_max

        xx_BC, zz_BC = torch.meshgrid(x_BC, z_BC, indexing='xy')
        yy_0 = torch.full_like(xx_BC, 0.0 + (offset / self.height))
        yy_1 = torch.full_like(xx_BC, 1.0 - (offset / self.height))

        tensor0 = torch.stack([xx_BC, yy_0, zz_BC], dim=-1).reshape(-1, 3)
        tensor1 = torch.stack([xx_BC, yy_1, zz_BC], dim=-1).reshape(-1, 3)

        yy_BC, zz_BC = torch.meshgrid(y_BC, z_BC, indexing='xy')
        xx_0 = torch.full_like(yy_BC, 0.0 + (offset / self.width))
        xx_1 = torch.full_like(yy_BC, 1.0 - (offset / self.width))

        tensor2 = torch.stack([xx_0, yy_BC, zz_BC], dim=-1).reshape(-1, 3)
        tensor3 = torch.stack([xx_1, yy_BC, zz_BC], dim=-1).reshape(-1, 3)

        tensor_array_BC_all = torch.cat([tensor0, tensor1, tensor2, tensor3], dim=0)
        self.register_buffer('tensor_array_BC_all', tensor_array_BC_all)

    def _init_bc_buffer_hash(self) :
        """
        Init all elements that will be used by Forward_BC in order
        to optimize time during training

        The difference with forward_physics is that all coordinate values are 
        normalize with the maximum value between x_max, y_max and z_max.

        This is a requierment to use hash encoding.
        """

        self.z_norm = self._normalize_z_for_nerf(self.z_values)

        x_min = self.x_norm[0].item()
        x_max = self.x_norm[-1].item()

        y_min = self.y_norm[0].item()
        y_max = self.y_norm[-1].item()

        xx_BC, zz_BC = torch.meshgrid(self.x_norm, self.z_norm, indexing='xy')
        yy_0 = torch.full_like(xx_BC, y_min)
        yy_1 = torch.full_like(xx_BC, y_max)

        self.register_buffer('tensor_array_BC_0', torch.stack([xx_BC, yy_0, zz_BC], dim=-1).reshape(-1, 3))
        self.register_buffer('tensor_array_BC_1', torch.stack([xx_BC, yy_1, zz_BC], dim=-1).reshape(-1, 3))

        yy_BC, zz_BC = torch.meshgrid(self.y_norm, self.z_norm, indexing='xy')
        xx_0 = torch.full_like(yy_BC, x_min)
        xx_1 = torch.full_like(yy_BC, x_max)

        self.register_buffer('tensor_array_BC_2',torch.stack([xx_0, yy_BC, zz_BC], dim=-1).reshape(-1, 3))
        self.register_buffer('tensor_array_BC_3',torch.stack([xx_1, yy_BC, zz_BC], dim=-1).reshape(-1, 3))
    
    def forward_physics(self, U_z0):
        """
        Call nerf and physics model to predict
        object at each coordinate and simulate 
        light propagation.

        Can be used only with positional_encoding and 
        positional_encoding_barf

        U_z0 : Target hologram to reconstruct
        """

        loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        loss_sparsity = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        complex_i = torch.complex(self.scalar_zero, self.scalar_one)
        phase_shift_complex = torch.complex(self.phase_shift, self.scalar_zero)

        U_z_following_prop = torch.complex(self.zeros_grid, self.zeros_grid)
        densities_1d = self.Nerf(self.coords_3d)

        volume_3d = densities_1d.squeeze(-1).view(len(self.z_values)+1, self.width, self.height)

        real_following_ref = self.ones_grid * self.incident_light
        U_z_following_ref = torch.complex(real_following_ref, self.zeros_grid)

        self.Physics_model.update_kernel(self.waveLength)
        
        volume_3d_complex = torch.complex(volume_3d, self.zeros_grid)
        full_phase_delay = torch.exp(complex_i * phase_shift_complex * volume_3d_complex)
        
        
        if self.with_sparsity:
            
            # --- FREE SPACE PRIOR (Optimize the Unseen, NeurIPS 2025) ---
            flat_volume = volume_3d.view(-1)
            N_samples = flat_volume.numel() // 10
            random_indices = torch.randint(0, flat_volume.numel(), (N_samples,), device=self.device)
            sampled_densities = flat_volume[random_indices]
            loss_sparsity = torch.mean(torch.square(torch.sigmoid(sampled_densities * 50.0)))
            
            # --- RAY ENTROPY LOSS (Compression sur l'axe Z) ---
            eps = 1e-8
            # 1. On s'assure que les valeurs sont positives
            vol_abs = torch.abs(volume_3d) + eps 
            
            # 2. On fait la somme des densités le long de l'axe Z (dim=0)
            # Ça nous donne le "poids total" de matière pour chaque pixel (X, Y)
            z_sum = torch.sum(vol_abs, dim=0, keepdim=True)
            
            # 3. On normalise pour créer une "probabilité" de présence sur l'axe Z
            # p_z sera proche de 1 si tout est au même endroit, et proche de 0.01 si c'est étalé
            prob_z = vol_abs / z_sum
            
            # 4. Calcul de l'Entropie : - p * log(p)
            entropy = - prob_z * torch.log(prob_z + eps)
            
            # 5. On somme l'entropie sur l'axe Z pour chaque rayon, puis on fait la moyenne globale
            ray_entropy = torch.sum(entropy, dim=0)
            loss_sparsity = torch.mean(ray_entropy)
        # --------------------------------------------------
            # ----------------------------------------------------------------

            
        weighted_loss_sparsity = self.sparsity_weight * loss_sparsity
        
        for i, z in enumerate(self.z_values):
            object_following = volume_3d[i]
            object_preceding = volume_3d[i+1]
            phase_delay = full_phase_delay[i+1]

            if i == 0 : 
                
                U_z_following_ref *= phase_delay

                U_z_following_prop = self.Physics_model.angular_spectrum_propagator(image=U_z_following_ref)

                loss += 0.5*torch.mean(torch.square(object_following))

            elif i==len(self.z_values)-1 :

                U_z_following_prop *= phase_delay

                U_z_following_prop = self.Physics_model.angular_spectrum_propagator(image=U_z_following_prop)
                
                U_z_following_prop_intensity = torch.square(torch.abs(U_z_following_prop))

                loss += torch.mean(torch.square(U_z0 - U_z_following_prop_intensity))
                loss += 0.5*torch.mean(torch.square(object_preceding))

            else:
                U_z_following_prop *= phase_delay

                U_z_following_prop = self.Physics_model.angular_spectrum_propagator(image=U_z_following_prop)
        
        loss+=weighted_loss_sparsity
        return loss, weighted_loss_sparsity, volume_3d
    
    def forward_BC(self) : 
        """
        Bundary conditions regularization.
        Add loss if any object is palaced at a bundary.
        """
        loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        object_BC_all = self.Nerf(self.tensor_array_BC_all)
        loss += 2.0 * torch.mean(torch.square(object_BC_all))
        return loss
    
    @torch.no_grad()
    def generate_output(self, save_dir) :
        """
        Generate files useful for visualization of 
        positional encoding and positional encoding barf based models.

        save_dir : directory to save files.
        """
        obj_dir = os.path.join(save_dir, 'obj')
        intensity_dir = os.path.join(save_dir, 'intensity')

        if os.path.exists(obj_dir):
            shutil.rmtree(obj_dir)
        if os.path.exists(intensity_dir):
            shutil.rmtree(intensity_dir)

        os.makedirs(obj_dir, exist_ok=True)
        os.makedirs(intensity_dir, exist_ok=True)

        print(f"Generating results on {self.device}...")

        # ======================================================================
        # PARTIE 1 : Reconstruct 3D Object (NeRF Query)
        # ======================================================================
        print("Reconstructing 3D Object slices...")
        volume_shape = (self.width, self.height, len(self.z_values))
        obj_volume = np.zeros(volume_shape, dtype=np.float32)
        
        densities_1d = self.Nerf(self.coords_3d)

        volume_3d = densities_1d.squeeze(-1).view(len(self.z_values)+1, self.width, self.height)

        obj_volume = volume_3d[:-1].permute(1, 2, 0).cpu().numpy()
        
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
        
        self.Physics_model.update_kernel(self.waveLength)

        volume_3d_complex = torch.complex(volume_3d, self.zeros_grid)
        full_phase_delay = torch.exp(complex_i * phase_shift_complex * volume_3d_complex)
        for i, z in enumerate(self.z_values):
            
            phase_delay = full_phase_delay[i]

            if i==0 : 
                U_z_following_ref_intensity = torch.square(torch.abs(U_z_following_ref)).cpu().numpy()
                U_z_following_ref_intensity = Image.fromarray(U_z_following_ref_intensity)
                U_z_following_ref_intensity.save(os.path.join(intensity_dir, f"Intensity_MorpHoloNet_{z.item():.1f}.tif"))
            
            elif i==1:           

                U_z_following_prop = U_z_following_ref * phase_delay


                U_z_following_prop = self.Physics_model.angular_spectrum_propagator(image=U_z_following_prop)
                
                U_z_following_prop_intensity = torch.square(torch.abs(U_z_following_prop)).cpu().numpy()
                U_z_following_prop_intensity = Image.fromarray(U_z_following_prop_intensity)
                U_z_following_prop_intensity.save(os.path.join(intensity_dir, f"Intensity_MorpHoloNet_{z.item():.1f}.tif"))


            elif i==len(self.z_values)-1:                

                U_z_following_prop *= phase_delay


                U_z_following_prop = self.Physics_model.angular_spectrum_propagator(image=U_z_following_prop)

                U_z_following_prop_intensity = torch.square(torch.abs(U_z_following_prop)).cpu().numpy()
                U_z_following_prop_intensity = Image.fromarray(U_z_following_prop_intensity)
                U_z_following_prop_intensity.save(os.path.join(intensity_dir, f"Intensity_MorpHoloNet_{z.item():.1f}.tif"))

                U_z_following_prop = self.Physics_model.angular_spectrum_propagator(image=U_z_following_prop)

                U_z_following_prop_intensity = torch.square(torch.abs(U_z_following_prop)).cpu().numpy()
                U_z_following_prop_intensity = Image.fromarray(U_z_following_prop_intensity)
                U_z_following_prop_intensity.save(os.path.join(intensity_dir, f"Intensity_MorpHoloNet_0.tif"))

            else:

                U_z_following_prop *= phase_delay


                U_z_following_prop = self.Physics_model.angular_spectrum_propagator(image=U_z_following_prop)

                U_z_following_prop_intensity = torch.square(torch.abs(U_z_following_prop)).cpu().numpy()
                U_z_following_prop_intensity = Image.fromarray(U_z_following_prop_intensity)
                U_z_following_prop_intensity.save(os.path.join(intensity_dir, f"Intensity_MorpHoloNet_{z.item():.1f}.tif"))

        print("Generation done")

    @torch.no_grad()
    def reconstruct_hologram(self) :
        """
        Generate reconstruct hologram of
        positional encoding and positional encoding barf 
        based models.
        """

        complex_i = torch.complex(self.scalar_zero, self.scalar_one)
        U_z_following_prop = torch.complex(self.zeros_grid, self.zeros_grid)
        real_following_ref = self.ones_grid * self.incident_light
        U_z_following_ref = torch.complex(real_following_ref, self.zeros_grid)
        phase_shift_complex = torch.complex(self.phase_shift, self.scalar_zero)

        densities_1d = self.Nerf(self.coords_3d)
        volume_3d = densities_1d.squeeze(-1).view(len(self.z_values)+1, self.width, self.height)
        
        self.Physics_model.update_kernel(self.waveLength)

        volume_3d_complex = torch.complex(volume_3d, self.zeros_grid)
        full_phase_delay = torch.exp(complex_i * phase_shift_complex * volume_3d_complex)

        for i, z in enumerate(self.z_values):         
            if i==0 :
                continue
            phase_delay = full_phase_delay[i]
            
            if i==1:
                U_z_following_prop = U_z_following_ref * phase_delay

                U_z_following_prop = self.Physics_model.angular_spectrum_propagator(image=U_z_following_prop)


            elif i==len(self.z_values)-1:
                U_z_following_prop *= phase_delay

                U_z_following_prop = self.Physics_model.angular_spectrum_propagator(image=U_z_following_prop)

                U_0_following_prop = self.Physics_model.angular_spectrum_propagator(image=U_z_following_prop)
        
            else:
                U_z_following_prop *= phase_delay

                U_z_following_prop = self.Physics_model.angular_spectrum_propagator(image=U_z_following_prop)
        
        return torch.square(torch.abs(U_0_following_prop))

    '''
    NOTE :
    Pour l'instant on suppose que l'application de bundary conditions est obligatoire
    Ainsi on ne met pas d'autres régularisation pour z_max et z_min
    Pour l'étude de bactéries proche des bords il sera peut être interessant de enlever le BC et 
    de le remplacer par les autres régularisations 
    '''
    def forward_physics_hash(self, U_z0):

        """
        NOTE: This method is not maintained.

        Call nerf and physics model to predict
        object at each coordinate and simulate 
        light propagation.
        Can be used only with hash_grid encoder.

        U_z0 : Target hologram to reconstruct
        """
        device = self.scalar_zero.device

        loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        loss_sparsity = torch.tensor(0.0, dtype=torch.float32, device=device)
        loss_tv = torch.tensor(0.0, dtype=torch.float32, device=device)
        loss_bc = torch.tensor(0.0, dtype=torch.float32, device=device)

        complex_i = torch.complex(self.scalar_zero, self.scalar_one)

        U_z_following_prop = torch.complex(self.zeros_grid, self.zeros_grid)

        for z in self.z_values:
            if torch.isclose(z, torch.tensor(self.z_max, dtype=torch.float32, device=device)):
                z_norm_following = self._normalize_z_for_nerf(z)
                z_following = torch.full_like(self.xx, z_norm_following.item())

                z_norm_preceding = self._normalize_z_for_nerf(z-self.dz)
                z_preceding = torch.full_like(self.xx, z_norm_preceding.item())

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

                if self.L2 : 
                    loss_bc += 0.5*torch.mean(torch.square(object_following))
                else :
                    loss_bc += torch.mean(torch.abs(object_following))

                if self.with_sparsity : 
                    loss_sparsity += torch.mean(torch.sqrt(torch.abs(object_preceding)+1e-8))

                if self.with_tv : 
                    tv_x = torch.mean(torch.abs(object_preceding[1:, :] - object_preceding[:-1, :]))
                    tv_y = torch.mean(torch.abs(object_preceding[:, 1:] - object_preceding[:, :-1]))
                    loss_tv += (tv_x + tv_y)

            elif torch.isclose(z, torch.tensor(self.z_min, dtype=torch.float32, device=device)):
                z_norm = self._normalize_z_for_nerf(z-self.dz)
                z_preceding = torch.full_like(self.xx, z_norm.item())
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

                if self.L2 : 
                    loss += torch.mean(torch.square(U_z0 - U_z_following_prop_intensity))
                    loss_bc += 0.5*torch.mean(torch.square(object_preceding))
                else :
                    loss += torch.mean(torch.square(U_z0 - U_z_following_prop_intensity))
                    loss_bc += torch.mean(torch.abs(object_preceding))

            else:
                z_norm = self._normalize_z_for_nerf(z - self.dz)
                z_preceding = torch.full_like(self.xx, z_norm.item())
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
                if self.with_sparsity : 
                    loss_sparsity += torch.mean(torch.sqrt(torch.abs(object_preceding)+1e-8))
                    
                if self.with_tv : 
                    tv_x = torch.mean(torch.abs(object_preceding[1:, :] - object_preceding[:-1, :]))
                    tv_y = torch.mean(torch.abs(object_preceding[:, 1:] - object_preceding[:, :-1]))
                    loss_tv += (tv_x + tv_y)

        weighted_loss_sparsity = self.sparsity_weight * loss_sparsity
        weighted_loss_tv = self.tv_weight * loss_tv
        
        loss += weighted_loss_sparsity
        loss += weighted_loss_tv
        
        return loss, weighted_loss_sparsity, weighted_loss_tv, loss_bc
    
    def _init_physics_buffer_hash(self) :
        '''
        Init all elements that will be used by forward_physics_hash in order
        to optimize time during training.

        The difference with forward_physics is that all coordinate values are 
        normalize with the maximum value between x_max, y_max and z_max.

        This is a requierment to use hash encoding.
        ''' 

        self.padding = 0.05
        self.scale = 1 - 2 * self.padding

        span_x = float(self.width)
        span_y = float(self.height)
        span_z = float(self.z_max-self.z_min)

        self.max_span = max(span_x, span_y, span_z)

        offset_x = (1.0 -(span_x/self.max_span))/2.0
        offset_y = (1.0 -(span_y/self.max_span))/2.0
        self.offset_z = (1.0 -(span_z/self.max_span))/2.0

        x_raw = torch.arange(0, self.width, dtype=torch.float32)
        y_raw = torch.arange(0, self.height, dtype=torch.float32)

        self.x_norm = ((x_raw/self.max_span) + offset_x) * self.scale + self.padding
        self.y_norm = ((y_raw/self.max_span) + offset_y) * self.scale + self.padding

        xx_norm, yy_norm = torch.meshgrid(self.x_norm, self.y_norm, indexing='xy')
        self.register_buffer('xx', torch.clamp(xx_norm, 0.0, 1.0 - 1e-5))
        self.register_buffer('yy', torch.clamp(yy_norm, 0.0, 1.0 - 1e-5))
        
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
    
    def _normalize_z_for_nerf(self, z_phys) : 

        """
        This method is used for hash encoding based methods
        Return the value of z normalized with the maximum
        value in x, y and z with padding.

        z_phys : value of int z plane
        """
        z_norm = (z_phys - self.z_min) / self.max_span
        z_centered = z_norm + self.offset_z
        z_final = z_centered * self.scale + self.padding
        return torch.clamp(z_final, 0.0, 1.0 - 1e-5)

    def forward_BC_hash(self) : 
            
            """
            NOTE: This method is not maintained.

            Bundary conditions regularization.
            Add loss if any object is palaced at a bundary.
            """
            device = self.scalar_zero.device
            loss = torch.tensor(0.0, dtype=torch.float32, device=device)

            object_BC_0 = self.Nerf(self.tensor_array_BC_0)
            object_BC_1 = self.Nerf(self.tensor_array_BC_1)

            if self.L2 : 
                loss += 0.5*torch.mean(torch.square(object_BC_0))
                loss += 0.5*torch.mean(torch.square(object_BC_1))
            else : 
                loss += torch.mean(torch.abs(object_BC_0))
                loss += torch.mean(torch.abs(object_BC_1))

            object_BC_2 = self.Nerf(self.tensor_array_BC_2)
            object_BC_3 = self.Nerf(self.tensor_array_BC_3)

            if self.L2 : 
                loss += 0.5*torch.mean(torch.square(object_BC_2))
                loss += 0.5*torch.mean(torch.square(object_BC_3))
            else : 
                loss += torch.mean(torch.abs(object_BC_2))
                loss += torch.mean(torch.abs(object_BC_3))

            return loss

    @torch.no_grad()
    def generate_output_hash(self, save_dir) : 
        """
        NOTE: This method is not maintained.

        Generate files useful for visualization of 
        hash encoding based models.

        save_dir : directory to save files.
        """
        obj_dir = os.path.join(save_dir, 'obj')
        intensity_dir = os.path.join(save_dir, 'intensity')
        
        if os.path.exists(obj_dir):
            shutil.rmtree(obj_dir)
        if os.path.exists(intensity_dir):
            shutil.rmtree(intensity_dir)

        os.makedirs(obj_dir, exist_ok=True)
        os.makedirs(intensity_dir, exist_ok=True)

        device = self.scalar_zero.device
        print(f"Generating results on {device}...")

        def get_z_norm(z_phys) : 
            if self.hash: 
                val = self._normalize_z_for_nerf(z_phys)
            else : 
                val = z_phys / self.z_max
            return val.item()

        # ======================================================================
        # PARTIE 1 : Reconstruct 3D Object (NeRF Query)
        # ======================================================================
        print("Reconstructing 3D Object slices...")
        volume_shape = (self.width, self.height, len(self.z_values))
        obj_volume = np.zeros(volume_shape, dtype=np.float32)
        
        for i, z in enumerate(self.z_values):
            z_norm = get_z_norm(z)
            z_filled = torch.full_like(self.xx, z_norm)

            tensor_array = torch.stack([self.xx, self.yy, z_filled], dim=-1).reshape(-1, 3)
            obj_slice = self.Nerf(tensor_array).reshape(self.width, self.height)
            obj_npy = obj_slice.cpu().numpy()

            obj_volume[:, :, i] = obj_npy

            #img = Image.fromarray(obj_npy)
            #img.save(os.path.join(obj_dir, f"obj_{z.item():.1f}.tif"))
        
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
                z_following = torch.full_like(self.xx, get_z_norm(z))
    
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
                z_following = torch.full_like(self.xx, get_z_norm(z))
    
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
                z_following = torch.full_like(self.xx, get_z_norm(z))
    
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
    def reconstruct_hologram_hash(self, U_z0) :

        """
        NOTE: This method is not maintained.

        Generate reconstruct hologram of
        hash encoding based models.

        U_z0 : Target hologram to reconstruct.
        """
        device = U_z0.device

        def get_z_norm(z_phys) : 
            if self.hash: 
                val = self._normalize_z_for_nerf(z_phys)
            else : 
                val = z_phys / self.z_max
            return val.item()

        complex_i = torch.complex(self.scalar_zero, self.scalar_one)
        U_z_following_prop = torch.complex(self.zeros_grid, self.zeros_grid)
        real_following_ref = self.ones_grid * self.incident_light
        U_z_following_ref = torch.complex(real_following_ref, self.zeros_grid)
        phase_shift_complex = torch.complex(self.phase_shift, self.scalar_zero)
        
        for z in self.z_values:
            if torch.isclose(z, torch.tensor(self.z_max, dtype=torch.float32, device=device)):
                continue
            
            elif torch.isclose(z, torch.tensor(self.z_max - self.dz, dtype=torch.float32, device=device)):
                z_following = torch.full_like(self.xx, get_z_norm(z))
    
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
                z_following = torch.full_like(self.xx, get_z_norm(z))
    
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
                z_following = torch.full_like(self.xx, get_z_norm(z))
    
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
    
    def get_internal_values(self):
        """
        Get phase_shift and incident_light values
        """
        return self.phase_shift.item(), self.incident_light.item()
    
    def update_barf_progress(self, current_epoch, barf_max_epochs):
        """
        Update the alpha parameter of Barf positional encoding.

        current_epoch : epoch currently ongoing.
        barf_max_epochs: max epochs with using barf.
        """
        progress = min(1.0, current_epoch/barf_max_epochs)
        for module in self.modules():
            if hasattr(module, "alpha_progress"):
                module.alpha_progress.fill_(progress)
                return progress 
        return 1.0

    def forward(self, U_z0):
            """
            Execute the training model based on provided parameters. 

            U_z0 : Target hologram to reconstruct.
            """
            loss_bc = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            volume_3d = None
            weighted_loss_sparsity = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            if self.hash :
                loss_physics, weighted_loss_sparsity, weighted_loss_tv, loss_bc_z = self.forward_physics_hash(U_z0)
                if self.with_bc :
                    loss_bc = self.forward_BC_hash()
                    loss_bc+=loss_bc_z
            else : 
                loss_physics, weighted_loss_sparsity, volume_3d = self.forward_physics(U_z0)
                if self.with_bc :
                    loss_bc = self.forward_BC()
                weighted_loss_tv = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            total_loss = loss_physics + loss_bc
            return loss_physics, loss_bc, weighted_loss_sparsity, weighted_loss_tv, total_loss, volume_3d