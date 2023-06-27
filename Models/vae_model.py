import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pickle
import pandas as pd
import os
from logger import logger
from logger import create_stats_ordered_dict
from Utils.utils import reparameterize
from Models.util_model import k_to_dts
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def layer_init(layer, weight_gain=1, bias_const=0, weights_init='xavier', bias_init='zeros'):
    
    if isinstance(layer, nn.Linear):
        if weights_init == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif weights_init == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        if bias_init == "zeros":
            torch.nn.init.constant_(layer.bias, bias_const)

LOG_STD_MAX = 0.0
LOG_STD_MIN = -3.0

# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(
        self, state_dim, action_dim, latent_dim, max_action, device, 
        hidden_size=750, policy_name=None, dep_check=False, dec_explo=False, loss_dyn=False,
        loss_energy=False, energy_coeff=1.,
        embed_size=0, daction_dim=0,
    ):
        super(VAE, self).__init__()
        self.device = device
        self.dep_check = dep_check  # discrete-continuous actions dependence check
        self.dec_explo = dec_explo  # decoder exploration: decode a latent action with a probability
        self.embed_size = embed_size
        self.daction_dim = daction_dim
        self.action_dim = action_dim
        self.loss_dyn = loss_dyn
        self.loss_energy = loss_energy
        

        # === encoder ===
        if not dep_check:
            self.e1 = nn.Linear(state_dim + action_dim, hidden_size)
            self.e2 = nn.Linear(hidden_size, hidden_size)
        else:
            self.e1 = nn.Linear(state_dim + action_dim-1, hidden_size)
            self.e2 = nn.Linear(hidden_size, hidden_size)
            self.e3 = nn.Linear(hidden_size, hidden_size)
            self.e4 = nn.Linear(1, hidden_size)
            # self.e1 = nn.Linear(state_dim + 1, hidden_size)
            # self.e2 = nn.Linear(hidden_size, hidden_size)
            # self.e3 = nn.Linear(action_dim - 1, hidden_size)
            # self.e4 = nn.Linear(hidden_size, hidden_size)


        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)


        # === decoder ===
        self.decoder_input_dim = state_dim + latent_dim
        self.decoder_output_dim = action_dim
        self.latent_disc_dim = 5

        if not self.dep_check:
            self.d1 = nn.Linear(self.decoder_input_dim, hidden_size)
            self.d2 = nn.Linear(hidden_size, hidden_size)
        else:
            self.d1 = nn.Linear(self.decoder_input_dim, hidden_size)
            self.d2 = nn.Linear(hidden_size, hidden_size)
        if not self.dec_explo:
            self.d3 = nn.Linear(hidden_size, self.decoder_output_dim)
        else:
            self.d3_mean = nn.Linear(hidden_size, self.decoder_output_dim)
            self.d3_std = nn.Linear(hidden_size, self.decoder_output_dim)

        if self.loss_dyn:
            self.d4 = nn.Linear(
                self.decoder_input_dim if not self.dep_check else hidden_size, 
                hidden_size
            )
            self.d5 = nn.Linear(hidden_size, hidden_size)
            self.d6 = nn.Linear(hidden_size, state_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim

    
    def forward(self, state, action):   # for discrete-continuous dependence
        if not self.dep_check:
            z = F.relu(self.e1(torch.cat([state, action], 1)))
            z = F.relu(self.e2(z))
        else:
            k = action[:, -1:]
            caction = action[:, :-1]
            z1 = F.relu(self.e1(torch.cat([state, caction], 1)))
            z1 = F.relu(self.e2(z1))
            z1 = F.relu(self.e3(z1))
            z2 = F.relu(self.e4(k))
            # z1 = F.relu(self.e1(torch.cat([state, k], 1)))
            # z1 = F.relu(self.e2(z1))
            # z2 = F.relu(self.e3(caction))
            # z2 = F.relu(self.e4(z2))
            z = z1 * z2

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        if self.dec_explo:
            u, state_diff, _ = self.decode(state, z=z)
        else:
            u, state_diff = self.decode(state, z=z)
        return u, mean, std, state_diff


    def decode(self, state,  z=None, clip=None, raw=False):
        state_ = state
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state_.shape[0], self.latent_dim)).to(self.device)
            if clip is not None:
                z = z.clamp(-clip, clip)
        
        if not self.dep_check:
            state_z = torch.cat([state_, z], 1)
            a = F.relu(self.d1(state_z))
            a = F.relu(self.d2(a))
        else:
            # a1 = F.relu(self.d1(state_))
            # a2 = F.relu(self.d2(z))
            # a = a1 * a2
            state_z = torch.cat([state_, z], 1)
            a = F.relu(self.d1(state_z))
            a = F.relu(self.d2(a))

        # === for dynamic loss ===
        s_diff = None
        if self.loss_dyn:
            if not self.dep_check:
                s_diff = F.relu(self.d4(state_z))
            else:
                s_diff = F.relu(self.d4(a))
            s_diff = F.relu(self.d5(s_diff))
            s_diff = self.d6(s_diff)


        
        # === for decoder exploration ===
        log_pi = None
        if not self.dec_explo:
            a = self.d3(a)
            action = self.max_action * torch.tanh(a)
            return action, s_diff
        else:
            mean_a = torch.tanh(self.d3_mean(a))
            log_std_a = torch.tanh(self.d3_std(a))
            log_std_a = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std_a + 1)  # From SpinUp / Denis Yarats
            action, log_pi = reparameterize(mean_a, log_std_a)
            return action, s_diff, log_pi


class VAEModule(object):
    def __init__(self, *args, vae_lr=1e-4, **kwargs):
        # print(f"[VAEModule] args:{args} | kwargs:{kwargs}")
        self.device = kwargs['device']
        self.energy_coeff = kwargs['energy_coeff']
        self.vae = VAE(*args, **kwargs).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)

    def train(self, buffer, folder_name, rms=None, batch_size=100, iterations=500000):
        logs = {'vae_loss': [], 'recon_loss': [], 'kl_loss': [], 'dyn_loss': []}
        for i in range(iterations):
            vae_loss, recon_loss, KL_loss, dyn_loss = self.train_step(buffer, rms=rms, batch_size=batch_size)
            logs['vae_loss'].append(vae_loss)
            logs['recon_loss'].append(recon_loss)
            logs['kl_loss'].append(KL_loss)
            logs['dyn_loss'].append(dyn_loss)

            if (i + 1) % 50000 == 0:
                print(f"vae_loss:{vae_loss} | recon_loss:{recon_loss} | kl_loss:{KL_loss} | dyn_loss:{dyn_loss}")
                print('Itr ' + str(i+1) + ' Training loss:' + '{:.4}'.format(vae_loss))
                self.save('model_' + str(i+1), folder_name)
                # pickle.dump(logs, open(folder_name + "/vae_logs.p", "wb"))
        pd.DataFrame(logs).to_csv(os.path.join(folder_name, 'vae_logs.csv'), index=False)
        

        return logs

    def loss(self, state, action, next_state):
        # recon: [B, Dim_act]
        recon, mean, std, pred_state_diff = self.vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        
        dyn_loss = torch.zeros(1).to(self.device)
        if self.vae.loss_dyn:
            state_diff = next_state - state
            dyn_loss = F.mse_loss(pred_state_diff, state_diff)  # [scalar]
            # vae_loss = recon_loss + 0.5 * KL_loss + 0.5 * dyn_loss
            # vae_loss = recon_loss + KL_loss +  dyn_loss
            vae_loss = recon_loss + 0.5 * KL_loss + 0.5 * dyn_loss
            if self.vae.loss_energy:
                recon_d_action = recon[:, -1:]   # [B, 1]
                l2_norm_cons = torch.norm(((recon_d_action-1)/2), 2, dim=1)  # [B,]    : energy consumption
                energy_loss_cons = -torch.mean(l2_norm_cons)   # scalar
                l2_norm_save = torch.norm(((recon_d_action+1)/2), 2, dim=1)  # [B,]  : energy save
                energy_loss_save = torch.mean(l2_norm_save)   # scalar
                
                # vae_loss += 2. * self.energy_coeff * energy_loss
                vae_loss += self.energy_coeff * energy_loss_cons
        else:
            vae_loss = recon_loss + 0.5 * KL_loss 
        return vae_loss, recon_loss, KL_loss, dyn_loss

    def train_step(self, buffer, rms=None, batch_size=100):
        if len(buffer) < batch_size:
            return 0, 0, 0

        # dataset_size = len(dataset['observations'])
        # ind = np.random.randint(0, dataset_size, size=batch_size)
        # state = dataset['observations'][ind]
        # action = dataset['actions'][ind]
        state, action, next_state, reward, not_done = buffer.sample(batch_size)

        # print(f"state:{state}")
        if rms is not None:
            state = rms.normalize(state)
        
        vae_loss, recon_loss, KL_loss, dyn_loss = self.loss(state, action, next_state)
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        return vae_loss.cpu().data.numpy(), recon_loss.cpu().data.numpy(), KL_loss.cpu().data.numpy(), dyn_loss.cpu().data.numpy()

    def save(self, filename, directory):
        torch.save(self.vae.state_dict(), '%s/%s_vae.pth' % (directory, filename))

    def load(self, filename, directory):
        self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename), map_location=self.device))
