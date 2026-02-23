import torch
import torch.nn as nn 
from .positional_encoding import PositionalEncoding
try:
    import tinycudann as tcnn
except ImportError:
    print("Erreur : tinycudann n'est pas installé.")
    exit()


class HoloTrack(nn.Module):
    def __init__(self, params):
        super(HoloTrack, self).__init__()

        self.encoder = PositionalEncoding(gaussian_projection=params["gaussian_proj"], gaussian_scale=params["gaussian_scale"]) # Gaussian_scale: 5~12
        
        self.model = tcnn.Network(
            n_input_dims = params["gaussian_proj"]*4, 
            n_output_dims=1,
            network_config={
                    "otype": "FullyFusedMLP", # Le mode Turbo de NVIDIA
                    "activation": "ReLU",     # ReLU est plus rapide que SiLU ici
                    "output_activation": "Sigmoid", # Force la sortie entre 0 et 1
                    "n_neurons": 128,          # 64 est optimal pour la vitesse sur RTX 3080
                    "n_hidden_layers": 3      # 2 ou 3 couches suffisent grâce au HashGrid
                }
        
        )
    def forward(self, pos):
        encoded = self.encoder(pos)
        encoded = encoded.to(torch.float32)
        output = self.model(encoded)
        return output.type(dtype=torch.float32)
