"""
Based on https://github.com/sfujim/BCQ
"""
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

def layer_init(layer, weight_gain=1, bias_const=0, weights_init='xavier', bias_init='zeros'):
    
    if isinstance(layer, nn.Linear):
        if weights_init == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif weights_init == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        if bias_init == "zeros":
            torch.nn.init.constant_(layer.bias, bias_const)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        # self.hidden_size = (512, 256, 256)

        # self.l1 = nn.Linear(state_dim, self.hidden_size[0])
        # self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        # self.l3 = nn.Linear(self.hidden_size[1], self.hidden_size[2])
        # self.l4 = nn.Linear(self.hidden_size[2], action_dim)

        self.hidden_size = (400, 300)

        self.l1 = nn.Linear(state_dim, self.hidden_size[0])
        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], action_dim)

        self.max_action = max_action

    def forward(self, state):
        # a = F.relu(self.l1(state))
        # a = F.relu(self.l2(a))
        # a = F.relu(self.l3(a))

        # return self.max_action * torch.tanh(self.l4(a))

        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))
LOG_STD_MAX = 0.0
LOG_STD_MIN = -3.0
class ActorSac(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, device):
        super().__init__()

        self.hidden_size = (512, 256, 256)

        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.mean = nn.Linear(128, action_dim)
        self.logstd = nn.Linear(128, action_dim)
        self.apply(layer_init)
        self.device = device

        self.max_action = max_action

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

        action,log_pi = reparameterize(mean, log_std)

        # === deterministic latent ===
        # return mean, log_std
        # === probabilitistic latent ===
        return action, log_pi

class ActorPerturbation(nn.Module):
    def __init__(self, state_dim, action_dim, latent_action_dim, max_action, max_latent_action=2, phi=0.05):
        super(ActorPerturbation, self).__init__()

        self.hidden_size = (512, 256, 256, 512, 256, 256)

        self.l1 = nn.Linear(state_dim, self.hidden_size[0])
        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], self.hidden_size[2])
        self.l4 = nn.Linear(self.hidden_size[2], latent_action_dim)

        self.l5 = nn.Linear(state_dim + action_dim, self.hidden_size[3])
        self.l6 = nn.Linear(self.hidden_size[3], self.hidden_size[4])
        self.l7 = nn.Linear(self.hidden_size[4], self.hidden_size[5])
        self.l8 = nn.Linear(self.hidden_size[5], action_dim)

        self.max_latent_action = max_latent_action
        self.max_action = max_action
        self.phi = phi

    def forward(self, state, decoder):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        latent_action = self.max_latent_action * torch.tanh(self.l4(a))

        mid_action = decoder(state, z=latent_action)

        a = F.relu(self.l5(torch.cat([state, mid_action], 1)))
        a = F.relu(self.l6(a))
        a = F.relu(self.l7(a))
        a = self.phi * torch.tanh(self.l8(a))
        final_action = (a + mid_action).clamp(-self.max_action, self.max_action)
        return latent_action, mid_action, final_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # self.hidden_size = (512, 256, 256)

        # self.l1 = nn.Linear(state_dim + action_dim, self.hidden_size[0])
        # self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        # self.l3 = nn.Linear(self.hidden_size[1], self.hidden_size[2])
        # self.l4 = nn.Linear(self.hidden_size[2], 1)

        # self.l5 = nn.Linear(state_dim + action_dim, self.hidden_size[0])
        # self.l6 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        # self.l7 = nn.Linear(self.hidden_size[1], self.hidden_size[2])
        # self.l8 = nn.Linear(self.hidden_size[2], 1)

        # self.hidden_size = (400, 300)
        self.hidden_size = (256, 256)

        self.l1 = nn.Linear(state_dim + action_dim, self.hidden_size[0])
        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], 1)

        self.l4 = nn.Linear(state_dim + action_dim, self.hidden_size[0])
        self.l5 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l6 = nn.Linear(self.hidden_size[1], 1)

    def forward(self, state, action):
        # q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        # q1 = F.relu(self.l2(q1))
        # q1 = F.relu(self.l3(q1))
        # q1 = self.l4(q1)

        # q2 = F.relu(self.l5(torch.cat([state, action], 1)))
        # q2 = F.relu(self.l6(q2))
        # q2 = F.relu(self.l7(q2))
        # q2 = self.l8(q2)
        # return q1, q2

        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        # q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        # q1 = F.relu(self.l2(q1))
        # q1 = F.relu(self.l3(q1))
        # q1 = self.l4(q1)
        # return q1

        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(
        self, state_dim, action_dim, latent_dim, max_action, device, 
        hidden_size=750, policy_name=None, dep_check=False, dec_explo=False, loss_dyn=False,
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
        recon, mean, std, pred_state_diff = self.vae(state, action)

        recon_loss = F.mse_loss(recon, action)
        
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        
        dyn_loss = torch.zeros(1).to(self.device)
        if self.vae.loss_dyn:
            state_diff = next_state - state
            dyn_loss = F.mse_loss(pred_state_diff, state_diff)
            vae_loss = recon_loss + 0.5 * KL_loss + 0.5 * dyn_loss
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


class Latent(object):
    def __init__(self, vae, state_dim, action_dim, latent_dim, max_action, device, discount=0.99, tau=0.005,
                 actor_lr=1e-3, critic_lr=1e-3, lmbda=0.75, max_latent_action=2, daction_dim=0,
                 nstep_return=False, min_period=0, max_period=0, dtdiscount=0.99, novae=False,
                 **kwargs):
        self.device = device
        self.policy_name = kwargs['policy_name']
        self.dec_explo = kwargs['dec_explo']
        self.nstep_return = nstep_return
        self.min_period = min_period
        self.max_period = max_period
        self.novae = novae


        latent_output_dim = latent_dim
        if self.policy_name == 'ddpg':
            self.actor = Actor(state_dim, latent_output_dim, max_latent_action).to(device)
        elif self.policy_name == 'sac':
            self.actor = ActorSac(state_dim, latent_output_dim, max_latent_action, device).to(device)
            autotune = True
        else:
            raise NotImplementedError
        # self.actor = Actor(state_dim, latent_dim, max_latent_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.loss_fn = nn.MSELoss()

            
        if self.policy_name == 'sac':
            # Automatic entropy tuning
            if autotune:
                # target_entropy = -float(out_c)
                self.target_entropy = -0.25
                self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
                self.alpha = self.log_alpha.exp().detach().cpu().item()
                self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)

                # target_entropy_d = -0.98 * np.log(1/out_d)
                self.target_entropy_d = -0.25
                self.log_alpha_d = torch.zeros(1, requires_grad=True, device=device)
                self.alpha_d = self.log_alpha_d.exp().detach().cpu().item()
                self.a_d_optimizer = torch.optim.Adam([self.log_alpha_d], lr=1e-4)
            else:
                self.alpha = 0.2
                self.alpha_d = 0.2

        self.latent_dim = latent_dim
        self.vae = vae
        self.max_action = max_action
        self.max_latent_action = max_latent_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.learning_step_train = 0
        self.dtdiscount = dtdiscount

    def select_action(self, state, latent_explor=False):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            if self.policy_name == 'ddpg':
                z = self.actor(state)
            elif self.policy_name == 'sac':
                z, _ = self.actor.get_action(state)
            # action = self.vae.decode(state, z=z)
            
            # === random exploration on latent space ===
            if latent_explor:
                z = torch.zeros_like(z).uniform_(-1, 1)
            
            if not self.novae:
                if self.dec_explo:
                    action, _, _ = self.vae.decode(state, z=z)
                    # print(f"action:{action}")
                else:
                    action, _ = self.vae.decode(state, z=z)
            else:
                action = copy.deepcopy(z)
        
        return action.cpu().data.numpy().flatten(), z.cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, rms=None, batch_size=100):
        for it in range(iterations):
            self.learning_step_train += 1
            # Sample replay buffer / batch
            if self.nstep_return:
                # state [L, B, D_state] | action [L ,B, D_action]
                state, action, next_state, reward, not_done = replay_buffer.sample_traj(batch_size, L=3)
            else:
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
                # print(f'action:{action}')
            # print(f"state:{state.shape} | action:{action.shape} | next_state:{next_state.shape} | reward:{reward.shape}")
            # print(f"action:{action[0]} | reward:{reward[0]}")
            if rms is not None:
                state = rms.normalize(state)
                next_state = rms.normalize(next_state)
            
            if self.policy_name == 'ddpg':
                critic_loss, actor_loss = self.ddpg_update(state, action, next_state, reward, not_done, self.learning_step_train)
            elif self.policy_name == 'sac':
                critic_loss, actor_loss = self.sac_update(state, action, next_state, reward, not_done, self.learning_step_train)
        return critic_loss, actor_loss
        
    def sac_update(self, state, action, next_state, reward, not_done, learning_step_train):
        # === calculate dt ===
        if self.nstep_return:
            state = state[0]
            next_state = next_state[-1]
            k = action[:, :, -1:].detach().cpu().numpy()  # [L, B, 1]
            action = action[0]

            dt = [torch.tensor(k_to_dts(k_, self.min_period, self.max_period), dtype=torch.int32, device=self.device) for k_ in k]
            dt = torch.stack(dt)    # [L, B, 1]
        else:
            k = action[:, -1:].detach().cpu().numpy() # [B, 1]
            dt = torch.tensor(k_to_dts(k, self.min_period, self.max_period), dtype=torch.int32, device=self.device)
            
        # === critic update ===
        with torch.no_grad():
            next_latent_action, next_latent_action_log_pi = self.actor.get_action(next_state)
            
            if not self.novae:
                if self.dec_explo:
                    next_action, _, next_action_log_pi = self.vae.decode(next_state, z=next_latent_action)
                    next_target_Q1, next_target_Q2 = self.critic_target(next_state, next_action)
                    next_target_Q = torch.min(next_target_Q1, next_target_Q2) - self.alpha * next_action_log_pi
                else:
                    next_action, _ = self.vae.decode(next_state, z=next_latent_action)
                    next_target_Q1, next_target_Q2 = self.critic_target(next_state, next_action)
                    next_target_Q = torch.min(next_target_Q1, next_target_Q2)
            else:
                next_target_Q1, next_target_Q2 = self.critic_target(next_state, next_latent_action)
                next_target_Q = torch.min(next_target_Q1, next_target_Q2) - self.alpha * next_latent_action_log_pi
                
            if self.nstep_return:
                next_target = self.get_nstep_return(reward, next_target_Q.squeeze(1), dt).unsqueeze(1)  # [B, 1]
            else:
                # next_target = reward + not_done * self.discount * next_target_Q
                next_target = (self.dtdiscount ** dt * reward) + not_done * self.discount * next_target_Q
                # print(f"reward:{reward.mean()} | dt_reward:{(self.discount**dt * reward).mean()}")
            
            # if learning_step_train % 1000 == 0:
            #     print(f"next_state:{next_state[0]} | next_target_Q:{next_target_Q[0]} | next_latent_action_log_pi:{next_latent_action_log_pi[0]}")
        
        current_Q1, current_Q2 = self.critic(state, action)
        Q1_loss = self.loss_fn(current_Q1, next_target)
        Q2_loss = self.loss_fn(current_Q2, next_target)
        critic_loss = (Q1_loss + Q2_loss) / 2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # print("======== check whether decoder state dictionay changes or not  =======")
        # for param in self.vae.state_dict():
        #     if 'd1' in param:
        #         print(f"{param}: {self.vae.state_dict()[param]}")
        
        # === actor update ===
        latent_action, latent_action_log_pi = self.actor.get_action(state)
        if not self.novae:
            if self.dec_explo:
                actor_action, _, action_action_log_pi = self.vae.decode(state, z=latent_action)
            else:
                actor_action, _ = self.vae.decode(state, z=latent_action)
        else:
            actor_action = latent_action
            action_action_log_pi = latent_action_log_pi
        actor_Q1, actor_Q2 = self.critic(state, actor_action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        if self.dec_explo:
            actor_loss = (self.alpha * action_action_log_pi - actor_Q).mean()
        else:
            actor_loss = -actor_Q.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # === soft update for target network ===
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss, actor_loss

    def get_nstep_return(self, rewards, final_value, dts, gamma=0.9):
        '''
        Input:
            rewards: [L, B,]
            final_value: [B, ]
            dts: [L, B, 1]
        '''
        # print("dts:{} | rewards:{}".format(dts.shape, rewards.shape))
        N = rewards.size()[0]
        # print("size N:", N)
        input_ = final_value
        # === typical return ===
        # for n in range(N-1, -1, -1):
        #     tmp_return = rewards[n] + gamma * input_
        #     input_ = tmp_return
        
        # === proposed return ===
        output_ = 0
        for n in range(N-1):
            output_ += (gamma ** dts[n]) * rewards[n]
            # output_ += (gamma ** n) * rewards[n]
        output_ += (gamma ** N) * final_value
        return output_


    def ddpg_update(self, state, action, next_state, reward, not_done, learning_step_train):
        # Critic Training
        with torch.no_grad():
            next_latent_action = self.actor_target(next_state)
            next_action, _ = self.vae.decode(next_state, z=next_latent_action)
            if learning_step_train % 1000 == 0:
                # print(f"next_state:{next_state[:5]}")
                # print(f"next_action:{next_action[:5]}")
                pass

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1 - self.lmbda) * torch.max(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
            # if learning_step_train % 1000 == 0:
            #     print(f"torch.max(target_Q1, target_Q2):{torch.mean(torch.max(target_Q1, target_Q2))}")

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Training
        latent_actions = self.actor(state)
        actions, _ = self.vae.decode(state, z=latent_actions)
        actor_loss = -self.critic.q1(state, actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Logging
        # logger.record_dict(create_stats_ordered_dict('Noise', noise.cpu().data.numpy(),))
        logger.record_dict(create_stats_ordered_dict('Q_target', target_Q.cpu().data.numpy(),))
        logger.record_tabular('Actor Loss', actor_loss.cpu().data.numpy())
        logger.record_tabular('Critic Loss', critic_loss.cpu().data.numpy())
        logger.record_dict(create_stats_ordered_dict('Actions', actions.cpu().data.numpy()))
        logger.record_dict(create_stats_ordered_dict('Latent Actions', latent_actions.cpu().data.numpy()))
        logger.record_dict(create_stats_ordered_dict('Latent Actions Norm', torch.norm(latent_actions, dim=1).cpu().data.numpy()))
        logger.record_dict(create_stats_ordered_dict('Current_Q', current_Q1.cpu().data.numpy()))
        assert (np.abs(np.mean(target_Q.cpu().data.numpy())) < 1e6)
        if learning_step_train % 1000 == 0:
            print(f"target_Q:{np.abs(np.mean(target_Q.cpu().data.numpy()))}")

        return critic_loss, actor_loss


    def save(self, filename, directory):
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))

        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))

    def load(self, filename, directory):
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/%s_critic_optimizer.pth' % (directory, filename)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        self.actor_target = copy.deepcopy(self.actor)


class LatentSac(object):
    def __init__(self, vae, state_dim, action_dim, latent_dim, max_action, device, discount=0.99, tau=0.005,
                 actor_lr=1e-3, critic_lr=1e-3, lmbda=0.75, max_latent_action=2, **kwargs):
        self.device = device
        self.actor = Actor(state_dim, latent_dim, max_latent_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.latent_dim = latent_dim
        self.vae = vae
        self.max_action = max_action
        self.max_latent_action = max_latent_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda


class LatentPerturbation(object):
    def __init__(self, vae, state_dim, action_dim, latent_dim, max_action, device, discount=0.99, tau=0.005,
                 actor_lr=1e-3, critic_lr=1e-3, lmbda=0.75, max_latent_action=2, phi=0.05, **kwargs):
        self.device = device
        
        self.actor = ActorPerturbation(state_dim, action_dim, latent_dim, max_action,
                                       max_latent_action=max_latent_action, phi=phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.vae = vae
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            _, _, action = self.actor(state, self.vae.decode)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100):

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # Critic Training
            with torch.no_grad():
                _, _, next_action = self.actor_target(next_state, self.vae.decode)
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1 - self.lmbda) * torch.max(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor Training
            latent_actions, mid_actions, actions = self.actor(state, self.vae.decode)
            actor_loss = -self.critic.q1(state, actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Logging
        logger.record_dict(create_stats_ordered_dict('Q_target', target_Q.cpu().data.numpy(),))
        logger.record_tabular('Actor Loss', actor_loss.cpu().data.numpy())
        logger.record_tabular('Critic Loss', critic_loss.cpu().data.numpy())
        logger.record_dict(create_stats_ordered_dict('Actions', actions.cpu().data.numpy()))
        logger.record_dict(create_stats_ordered_dict('Mid Actions', mid_actions.cpu().data.numpy()))
        logger.record_dict(create_stats_ordered_dict('Latent Actions', latent_actions.cpu().data.numpy()))
        logger.record_dict(create_stats_ordered_dict('Latent Actions Norm', torch.norm(latent_actions, dim=1).cpu().data.numpy()))
        logger.record_dict(create_stats_ordered_dict('Perturbation Norm', torch.norm(actions-mid_actions, dim=1).cpu().data.numpy()))
        logger.record_dict(create_stats_ordered_dict('Current_Q', current_Q1.cpu().data.numpy()))
        assert (np.abs(np.mean(target_Q.cpu().data.numpy())) < 1e6)

    def save(self, filename, directory):
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))

        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))

    def load(self, filename, directory):
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/%s_critic_optimizer.pth' % (directory, filename)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        self.actor_target = copy.deepcopy(self.actor)
