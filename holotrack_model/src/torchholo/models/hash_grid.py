import torch
import torch.nn as nn 

#NOTE: This part is not maintained
"""
try:
    import tinycudann as tcnn
except ImportError:
    print("Erreur : tinycudann n'est pas installé.")
    exit()
"""

class Hash_Grid(nn.Module):
    def __init__(self):
        super(Hash_Grid, self).__init__()

        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims = 3, 
            n_output_dims=1,
            encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 10,         # Standard pour une bonne résolution
                    "n_features_per_level": 2,
                    "log2_hashmap_size": 14, # 2^19 paramètres max (ajuster selon VRAM)
                    "base_resolution": 16,
                    "per_level_scale": 1.5,
                },
            network_config={
                    "otype": "FullyFusedMLP", # Le mode Turbo de NVIDIA
                    "activation": "ReLU",     # ReLU est plus rapide que SiLU ici
                    "output_activation": "None", # Force la sortie entre 0 et 1
                    "n_neurons": 64,          # 64 est optimal pour la vitesse sur RTX 3080
                    "n_hidden_layers": 2      # 2 ou 3 couches suffisent grâce au HashGrid
                }
        
        )
    def forward(self, pos):
        output = self.model(pos)

        clamped_output = torch.clamp(output, min=0.0, max=1.0)
        return clamped_output.type(dtype=torch.float32)
