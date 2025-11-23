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
        self.output_layer = nn.Sequential(
            nn.Linear(64, 3*out_dim),
            nn.Sigmoid()  # Para que el valor de salida esté entre 0 y 1
        )

    def forward(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        act1 = F.relu(self.layer1(obs))
        act2 = F.relu(self.layer2(act1))
        out = self.output_layer(act2)
        # Dividir en 3 tensores de tamaño batch x 10
        if len(obs.shape) == 1:
            o1 = out[0:self.out_dim]
            o2 = o1 + F.relu(out[self.out_dim:2 * self.out_dim]) + 0.01  # o2 >= o1 elemento a elemento
            o3 = o2 + F.relu(out[2 * self.out_dim:3 * self.out_dim]) + 0.01  # o3 >= o2 elemento a elemento
        else:
            o1 = out[:,0,0:self.out_dim]
            o2 = o1 + F.relu(out[:,0,self.out_dim:2*self.out_dim])+0.01  # o2 >= o1 elemento a elemento
            o3 = o2 + F.relu(out[:,0,2*self.out_dim:3*self.out_dim])+0.01  # o3 >= o2 elemento a elemento
        # Concatenar en una única salida batch x 3 x outdim
        ordered_out = torch.cat([o1, o2, o3], dim=len(o1.shape)-1)
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
    def __init__(self, in_dim, out_dim):
        super(DecisionNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, membership):
        if isinstance(membership, np.ndarray):
            membership = torch.tensor(membership, dtype=torch.float)

        act = F.relu(self.layer1(membership))  # Función de activación ReLU en capa oculta
        act = F.relu(self.layer2(act))
        ret = self.layer3(act)
        return ret


class RedOrdenada(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.fc = nn.Linear(64, 3 * output_dim)  # salida sin restricciones

    def forward(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        out = self.layer1(obs)
        out = self.layer2(out)
        out = self.fc(out)
        out = out.view(-1, 3, out.size(-1) // 3)  # separar 3 grupos de salidas

        # Primera salida: base acotada entre 0 y 1
        y1 = torch.sigmoid(out[:, 0, :])

        # Incrementos positivos garantizados con softplus (f(x)=log(1+exp(x)) > 0)
        d1 = F.softplus(out[:, 1, :])
        d2 = F.softplus(out[:, 2, :])

        # Normalizar incrementos para que la suma total no pase de 1 - y1
        total_increment = d1 + d2
        factor = torch.clamp((0.9 - y1) / (total_increment + 1e-8), min=0.01, max=1.0)
        d1 = d1 * factor
        d2 = d2 * factor

        y2 = y1 + d1
        y3 = y2 + d2

        return torch.cat((y1, y2, y3), dim=1)