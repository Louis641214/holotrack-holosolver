import torch
import torch.nn as nn

class PhysicsModel(nn.Module) : 
    def __init__(self, segment_size, physicalLength, depth, device):
        super().__init__()
        M2 = segment_size
        N2 = segment_size
        self.device = device

        u = torch.fft.fftfreq(M2, d=1.0, device=device) * M2
        v = torch.fft.fftfreq(N2, d=1.0, device=device) * N2
        V, U = torch.meshgrid(v, u, indexing='xy')

        U_base = U / (M2 * physicalLength)
        V_base = V / (N2 * physicalLength)

        S2 = (U_base**2 + V_base**2).to(torch.float32)

        self.register_buffer("S2", S2)
                
        if isinstance(depth, torch.Tensor) : 
            depth_tensor = depth.to(torch.complex64)
        else : 
            depth_tensor = torch.tensor(depth, dtype=torch.complex64, device=self.device)
        
        self.register_buffer('depth_tensor', depth_tensor)
        
    
    def update_kernel(self, waveLength) :
        k = 2 * torch.pi / waveLength

        freq_sq = ((waveLength**2)*self.S2).to(torch.complex64)
        F = 1j * k * torch.sqrt(1.0 - freq_sq)

        self.G = torch.exp(self.depth_tensor * F)

    def angular_spectrum_propagator(self, image):
        if self.depth_tensor.item() == 0:
            return image.to(torch.complex64)
        
        H = torch.fft.fft2(image.to(torch.complex64))
        
        recons = torch.fft.ifft2(H * self.G)

        return recons
