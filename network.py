import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output


class OrderedOutputNN(nn.Module):
    """
        Crea una red neuronal cuyas 3 salidas están ordenadas para poder obtener
        triángulos coherentes.

        Parameters:
            in_dim: Tamaño de la entrada
            out_dim: Tamaño de la salida, pero multiplicado x 3 debido a p1, pc y p2

        Returns:
            None
    """
    def __init__(self, in_dim, out_dim):
        super(OrderedOutputNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 3*out_dim)

    def forward(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        if isinstance(obs["Accuracy"], float):
            obs = torch.tensor([obs["Accuracy"]], dtype=torch.float)

        act1 = F.relu(self.layer1(obs))
        act2 = F.relu(self.layer2(act1))
        out = self.layer3(act2)
        # Dividir en 3 tensores de tamaño batch x 10
        o1 = out[0:self.out_dim]
        o2 = o1 + F.relu(out[self.out_dim:2*self.out_dim])  # o2 >= o1 elemento a elemento
        o3 = o2 + F.relu(out[2*self.out_dim:3*self.out_dim])  # o3 >= o2 elemento a elemento
        # Concatenar en una única salida batch x 3 x outdim
        ordered_out = torch.cat([o1, o2, o3], dim=0)
        return ordered_out

class DecisionNN(nn.Module):
    """
        Crea la red del Actor encargado de clasificar (decidir).

        Parameters:
            in_dim: Tamaño de la entrada, pero hay que incorporar un ejemplo del batch x10
            out_dim: Tamaño de la salida binaria (2)

        Returns:
            None
    """
    def __init__(self, in_dim):
        super(DecisionNN, self).__init__()
        self.in_dim = in_dim
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, obs):
        act = F.relu(self.layer1(obs))  # Función de activación ReLU en capa oculta
        act = F.relu(self.layer2(act))
        ret = torch.sigmoid(self.output_layer3(act))  # Sigmoide para probabilidad binaria
        return ret