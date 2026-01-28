import torch
import numpy as np

def angular_spectrum_propagator(image, depth, device, segment_size, physicalLength = 0.5, waveLength = 0.532 / 1.333):
    if depth == 0:
        return image.to(torch.complex64)
    
    H = torch.fft.fft2(image.to(torch.complex64))
    M2 = segment_size
    N2 = segment_size
    k = 2 * np.pi / waveLength

    u = torch.fft.fftfreq(M2, d=1.0, device=device) * M2
    v = torch.fft.fftfreq(N2, d=1.0, device=device) * N2

    V, U = torch.meshgrid(v, u, indexing='xy')
    U = (waveLength * U / (M2 * physicalLength)).to(torch.float32)
    V = (waveLength * V / (N2 * physicalLength)).to(torch.float32)
    U2 = (U**2).to(torch.complex64)
    V2 = (V**2).to(torch.complex64)
    F = 1j * k * torch.sqrt(1 - U2 - V2)
    
    if isinstance(depth, torch.Tensor) : 
        depth = depth.to(torch.complex64)
    else : 
        depth = torch.tensor(depth, dtype=torch.complex64, device=device)

    G = torch.exp(depth * F)
    recons = torch.fft.ifft2(H * G)
    simulated_image = (recons[0:M2, 0:N2]).to(torch.complex64)

    return simulated_image
