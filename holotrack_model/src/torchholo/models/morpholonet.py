import torch.nn as nn
from .positional_encoding import PositionalEncoding
class MorpHoloNet(nn.Module):
    def __init__(self, params):
        super(MorpHoloNet, self).__init__()

        self.p = PositionalEncoding(gaussian_projection=params["gaussian_proj"], gaussian_scale=params["gaussian_scale"]) # Gaussian_scale: 5~12
        
        self.layers = nn.ModuleList([nn.Linear(in_features=256, out_features=128, bias=True), 
                                    nn.Linear(in_features=128, out_features=128, bias=True),
                                    nn.Linear(in_features=128, out_features=128, bias=True)])
        
        self.last_layer = nn.Linear(in_features=128, out_features = 1, bias=True)

        self.activator = nn.SiLU(inplace=True)
        self.last_activator  = nn.Sigmoid()

        self.init_weights()

    def init_weights(self) : 

        for layer in self.layers : 
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="linear")
            nn.init.constant_(layer.bias, 0)

        nn.init.xavier_uniform_(self.last_layer.weight)
        nn.init.constant_(self.last_layer.bias, 0)


    def forward(self, pos):
        pos = self.p(pos)
        
        for layer in self.layers : 
            pos = self.activator(layer(pos))

        pos = self.last_layer(pos)
        return self.last_activator(pos)
