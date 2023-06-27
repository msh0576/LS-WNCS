import torch
# from torch import nn
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

from MPN_project.delay_sac.network.initializer import initialize_weight
from MPN_project.delay_sac.utils import build_mlp, reparameterize


class GaussianPolicy(torch.jit.ScriptModule):
    """
    Policy parameterized as diagonal gaussian distribution.
    """

    def __init__(self, action_shape, num_sequences, feature_dim, hidden_units=(256, 256)):
        super(GaussianPolicy, self).__init__()

        # NOTE: Conv layers are shared with the latent model.
        self.net = build_mlp(
            input_dim=num_sequences * feature_dim + (num_sequences - 1) * action_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=nn.ReLU(inplace=False),
        ).apply(initialize_weight)

    @torch.jit.script_method
    def forward(self, feature_action):
        means = torch.chunk(self.net(feature_action), 2, dim=-1)[0]
        return torch.tanh(means)

    @torch.jit.script_method
    def sample(self, feature_action):
        mean, log_std = torch.chunk(self.net(feature_action), 2, dim=-1)
        # print(f"mean:{mean.shape} | log_std:{log_std.shape}")
        # action,log_pi = reparameterize(mean, log_std.clamp_(-20, 2))
        action,log_pi = reparameterize(mean, log_std)

        # std = log_std.exp()
        # normal = Normal(mean, std)
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # action = torch.tanh(x_t)
        # log_pi = normal.log_prob(x_t)
        # log_pi -= torch.log(1.0 - action.pow(2) + 1e-8)

        return action, log_pi



# class GaussianPolicy(nn.Module):
#     """
#     Policy parameterized as diagonal gaussian distribution.
#     """

#     def __init__(self, action_shape, num_sequences, feature_dim, hidden_units=(256, 256)):
#         super(GaussianPolicy, self).__init__()

#         # NOTE: Conv layers are shared with the latent model.
#         self.net = build_mlp(
#             input_dim=num_sequences * feature_dim + (num_sequences - 1) * action_shape[0],
#             output_dim=2 * action_shape[0],
#             hidden_units=hidden_units,
#             hidden_activation=nn.ReLU(inplace=True),
#         ).apply(initialize_weight)

#         # input_dim = num_sequences * feature_dim + (num_sequences - 1) * action_shape[0]
#         # output_dim = action_shape[0]
#         # self.fc1 = nn.Linear(input_dim, 512)
#         # self.fc2 = nn.Linear(512, 256)
#         # self.fc3 = nn.Linear(256, 128)
#         # self.fc4 = nn.Linear(128, 64)
#         # self.mean = nn.Linear(64, output_dim)
#         # self.logstd = nn.Linear(64, output_dim)
#         # self.apply(initialize_weight)

#         self.LOG_STD_MAX = 0.0
#         self.LOG_STD_MIN = -3.0

#     def forward(self, feature_action):
#         means = torch.chunk(self.net(feature_action), 2, dim=-1)[0]
#         return torch.tanh(means)

#         # x = F.leaky_relu(self.fc1(feature_action), 0.01)
#         # x = F.leaky_relu(self.fc2(x), 0.01)
#         # x = F.leaky_relu(self.fc3(x), 0.01)
#         # x = F.leaky_relu(self.fc4(x), 0.01)
#         # mean = torch.tanh(self.mean(x))
#         # log_std = torch.tanh(self.logstd(x))
#         # log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

#         # return mean, log_std
    
#     def sample(self, feature_action):
#         mean, log_std = torch.chunk(self.net(feature_action), 2, dim=-1)
#         # action, log_pi = reparameterize(mean, log_std.clamp_(-20, 2))

#         std = log_std.exp()
#         normal = Normal(mean, std)
#         x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         action = torch.tanh(x_t)
#         log_pi = normal.log_prob(x_t)
#         log_pi -= torch.log(1.0 - action.pow(2) + 1e-8)


#         # mean, log_std = self.forward(feature_action)
#         # std = log_std.exp()
#         # normal = Normal(mean, std)
#         # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         # action = torch.tanh(x_t)
#         # log_pi = normal.log_prob(x_t)
#         # log_pi -= torch.log(1.0 - action.pow(2) + 1e-8)
#         return action, log_pi

class TwinnedQNetwork(torch.jit.ScriptModule):
    """
    Twinned Q networks.
    """

    def __init__(
        self,
        action_shape,
        z1_dim,
        z2_dim,
        hidden_units=(256, 256),
    ):
        super(TwinnedQNetwork, self).__init__()

        self.net1 = build_mlp(
            input_dim=action_shape[0] + z1_dim + z2_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=nn.ReLU(inplace=True),
        ).apply(initialize_weight)
        self.net2 = build_mlp(
            input_dim=action_shape[0] + z1_dim + z2_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=nn.ReLU(inplace=True),
        ).apply(initialize_weight)

    @torch.jit.script_method
    def forward(self, z, action):
        x = torch.cat([z, action], dim=1)
        return self.net1(x), self.net2(x)
