import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

from Models.delay_sac.utils import reparameterize


# ALGO LOGIC: initialize agent here:
LOG_STD_MAX = 0.0
LOG_STD_MIN = -3.0

def layer_init(layer, weight_gain=1, bias_const=0, weights_init='xavier', bias_init='zeros'):
    
    if isinstance(layer, nn.Linear):
        if weights_init == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif weights_init == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        if bias_init == "zeros":
            torch.nn.init.constant_(layer.bias, bias_const)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        
class Policy_SAC(nn.Module):
    def __init__(self, input_shape, out_c, device):
        super(Policy_SAC, self).__init__()
        self.fc1 = nn.Linear(input_shape, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.mean = nn.Linear(64, out_c)
        self.logstd = nn.Linear(64, out_c)
        self.apply(layer_init)
        self.device = device

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.Tensor(x).to(self.device)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = F.leaky_relu(self.fc3(x), 0.01)
        x = F.leaky_relu(self.fc4(x), 0.01)
        mean = torch.tanh(self.mean(x))
        log_std = torch.tanh(self.logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self.forward(x)
        # std = log_std.exp()
        # normal = Normal(mean, std)
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # action_c = torch.tanh(x_t)
        # log_prob_c = normal.log_prob(x_t)
        # log_prob_c -= torch.log(1.0 - action_c.pow(2) + 1e-8)

        action,log_pi = reparameterize(mean, log_std)



        return action, log_pi

    def exploit(self, x):
        mean, log_std = self.forward(x)
        return mean, log_std

class SoftQNetwork(nn.Module):   # for SAC
    def __init__(self, input_shape, out_c, layer_init, device):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape + out_c, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.apply(layer_init)
        self.device = device

    def forward(self, x, action_c):
        x = torch.cat([x, action_c], 1)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = F.leaky_relu(self.fc3(x), 0.01)
        x = F.leaky_relu(self.fc4(x), 0.01)
        x = self.fc5(x)
        return x

