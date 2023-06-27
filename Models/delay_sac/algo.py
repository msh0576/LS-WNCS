import os

import numpy as np
import torch
from torch.optim import Adam

from MPN_project.delay_sac.buffer import ReplayBuffer
from MPN_project.delay_sac.network import GaussianPolicy, LatentModel, TwinnedQNetwork
from MPN_project.delay_sac.utils import create_feature_actions, grad_false, soft_update
from MPN_project.delay_sac.env import delay_step, get_delay, dt_step
from MPN_project.MPN_func import scheduler
from MPN_project.running_stats import RunningStats, preprocess_norm

# from slac_master.slac.buffer import ReplayBuffer
# from slac_master.slac.network import GaussianPolicy, LatentModel, TwinnedQNetwork
# from slac_master.slac.utils import create_feature_actions, grad_false, soft_update

class SlacAlgorithm:
    """
    Stochactic Latent Actor-Critic(SLAC).

    Paper: https://arxiv.org/abs/1907.00953
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        action_repeat,
        device,
        seed,
        encoder_type,
        decoder_type,
        use_image=True,
        pkt_loss=0.5,
        num_plant=7,
        gamma=0.99,
        batch_size_sac=256,
        batch_size_latent=32,
        buffer_size=10 ** 5,
        num_sequences=8,
        lr_sac=3e-4,
        lr_latent=1e-4,
        feature_dim=256,
        # feature_dim=5,
        z1_dim=32,
        z2_dim=256,
        hidden_units=(256, 256),
        tau=5e-3,
        obs_normal=False,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # running mean and std of observations
        self.rms = RunningStats(dim=state_shape[0], device=device) if obs_normal else None

        # Replay buffer.
        self.buffer = ReplayBuffer(buffer_size, num_sequences, state_shape, action_shape, device, use_image, self.rms)

        # Networks.
        print("[slac algo] state:{} | action:{} | num_sequences:{}".format(state_shape, action_shape, num_sequences))
        self.actor = GaussianPolicy(action_shape, num_sequences, feature_dim, hidden_units).to(device)
        self.critic = TwinnedQNetwork(action_shape, z1_dim, z2_dim, hidden_units).to(device)
        self.critic_target = TwinnedQNetwork(action_shape, z1_dim, z2_dim, hidden_units).to(device)
        self.latent = LatentModel(state_shape, action_shape, use_image, 
            encoder_type, decoder_type, feature_dim, z1_dim, z2_dim, hidden_units).to(device)
        soft_update(self.critic_target, self.critic, 1.0)
        grad_false(self.critic_target)

        # Target entropy is -|A|.
        self.target_entropy = -float(action_shape[0])
        # We optimize log(alpha) because alpha is always bigger than 0.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()

        # Optimizers.
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_sac)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_sac)
        self.optim_alpha = Adam([self.log_alpha], lr=lr_sac)
        self.optim_latent = Adam(self.latent.parameters(), lr=lr_latent)

        self.learning_steps_sac = 0
        self.learning_steps_latent = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_repeat = action_repeat
        self.device = device
        self.gamma = gamma
        self.batch_size_sac = batch_size_sac
        self.batch_size_latent = batch_size_latent
        self.num_sequences = num_sequences
        self.tau = tau
        self.schedule_size = num_plant+1

        # JIT compile to speed up.
        fake_feature = torch.empty(1, num_sequences + 1, feature_dim, device=device)
        fake_action = torch.empty(1, num_sequences, action_shape[0], device=device)
        self.create_feature_actions = torch.jit.trace(create_feature_actions, (fake_feature, fake_action))

        # network scheduler
        self.random_scheduler = scheduler(algo='sequential', schedule_len=num_plant+1)

        

        # Other parameters
        self.prev_state = None
        self.prev_action = None
        self.pkt_loss = pkt_loss
        self.num_plant = num_plant
        self.use_image = use_image
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

    def preprocess(self, ob):
        if self.use_image:
            state = torch.tensor(ob.state, dtype=torch.uint8, device=self.device).float().div_(255.0)
        else:
            if self.rms is not None:
                state = self.rms.unnormalize(ob.state)
            else:
                state = torch.tensor(ob.state, dtype=torch.float, device=self.device)   # [1, num_seq, D_state]

        with torch.no_grad():
            # print("self.latent.encoder(state):", self.latent.encoder(state).shape)  # [1, 8, 5(D_latent)]
            feature = self.latent.encoder(state).view(1, -1)    # [1, 40]
        action = torch.tensor(ob.action, dtype=torch.float, device=self.device) # [1, 7]
        feature_action = torch.cat([feature, action], dim=1)    # [1, 47]
        # print(f"state:{state.shape} | action:{action.shape} | feature:{feature.shape} | feature_action:{feature_action.shape}")
        # print(f"state:{state}")
        return feature_action

    def explore(self, ob):
        feature_action = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor.sample(feature_action)[0]
        # print(f"action:{action}")
        return action.cpu().numpy()[0]

    def exploit(self, ob):
        feature_action = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor(feature_action)
            # action = self.actor(feature_action)[0]
        return action.cpu().numpy()[0]

    def step(self, env, ob, t, is_random, delay_step_=False):
        t += 1
        # == select schedule ==
        dt = int(self.random_scheduler.get_period(syst_idx=0))

        # == select action ==
        if is_random:
            action = env.action_space.sample()
        else:
            action = self.explore(ob)

        if not delay_step_:
            # state, reward, done, _ = env.step(action)
            # print(f"action:{action} | dt:{dt}")
            state, reward, done, _ = dt_step(env, action, dt)
        else:
            # == action and observation delay
            act_delay =  min(get_delay(self.pkt_loss), dt-1)
            obs_delay =  min(get_delay(self.pkt_loss), dt-1)
            # == env step ==
            state, reward, done, _  = delay_step(action, self.prev_action, dt, act_delay, obs_delay)

        # mask = False if t == env._max_episode_steps else done
        mask = False if t == env.spec.max_episode_steps else done
        ob.append(state, action)
        self.buffer.append(action, reward, mask, state, done)
        self.prev_action = action
        # == state normalization prepare ==
        if self.rms is not None:
            self.rms += state 

        if done:
            t = 0
            state = env.reset()
            ob.reset_episode(state)
            self.buffer.reset_episode(state)
            self.prev_action = env.action_space.sample()
            if self.rms is not None:
                self.rms += state 

        return t, reward, done
    
    
    def evaluate_steps(self, env, ob, delay_step_=False):
        state = env.reset()
        ob.reset_episode(state)
        episode_return = 0.0
        done = False
        prev_action = env.action_space.sample()
        dt_ratio = [0 for _ in range(self.schedule_size)]

        while not done:
            dt = int(self.random_scheduler.get_period(syst_idx=0))
            action = self.exploit(ob)
            if not delay_step_:
                state, reward, done, _ = dt_step(env, action, dt)
            else:
                # == action and observation delay
                act_delay =  min(get_delay(self.pkt_loss), dt-1)
                obs_delay =  min(get_delay(self.pkt_loss), dt-1)
                # == env step ==
                state, reward, done, _  = delay_step(action, prev_action, dt, act_delay, obs_delay)
            ob.append(state, action)
            episode_return += reward
            prev_action = action
            dt_ratio[dt-1] += 1

        return episode_return, dt_ratio

    def update_latent(self, writer):
        self.learning_steps_latent += 1
        state_, action_, reward_, done_ = self.buffer.sample_latent(self.batch_size_latent)
        loss_kld, loss_image, loss_reward = self.latent.calculate_loss(state_, action_, reward_, done_)

        self.optim_latent.zero_grad()
        (loss_kld + loss_image + loss_reward).backward()
        self.optim_latent.step()

        if self.learning_steps_latent % 1000 == 0:
            # print(f"loss_reward:{loss_reward:.4f} | loss_image:{loss_image:.4f}")
            writer.add_scalar("loss/kld", loss_kld.item(), self.learning_steps_latent)
            writer.add_scalar("loss/reward", loss_reward.item(), self.learning_steps_latent)
            writer.add_scalar("loss/image", loss_image.item(), self.learning_steps_latent)

    def update_sac(self, writer):
        self.learning_steps_sac += 1
        state_, action_, reward, done = self.buffer.sample_sac(self.batch_size_sac)
        # state_ [B, T+1, D_image(3, 64, 64)] | action_: [B, T, D_action(1)]
        # print(f"shape| state_:{state_.shape} | action_:{action_.shape}")
        z, next_z, action, feature_action, next_feature_action = self.prepare_batch(state_, action_)

        self.update_critic(z, next_z, action, next_feature_action, reward, done, writer)
        self.update_actor(z, feature_action, writer)
        soft_update(self.critic_target, self.critic, self.tau)

    def prepare_batch(self, state_, action_):
        with torch.no_grad():
            # f(1:t+1)
            feature_ = self.latent.encoder(state_)  # [B, T+1, D_feature]
            # z(1:t+1)
            z_ = torch.cat(self.latent.sample_posterior(feature_, action_)[2:4], dim=-1)

        # z(t), z(t+1)
        z, next_z = z_[:, -2], z_[:, -1]
        # a(t)
        action = action_[:, -1]
        # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t))
        feature_action, next_feature_action = self.create_feature_actions(feature_, action_)

        return z, next_z, action, feature_action, next_feature_action

    def update_critic(self, z, next_z, action, next_feature_action, reward, done, writer):
        curr_q1, curr_q2 = self.critic(z, action)
        # print(f"action:{action.shape}")   # [B, 1]
        with torch.no_grad():
            next_action, log_pi = self.actor.sample(next_feature_action)
            # print(f"next_action:{next_action.shape} | log_pi:{log_pi.shape}") # [B, D_action]
            # print(f"next_z:{next_z.shape}") # [B, 288]
            next_q1, next_q2 = self.critic_target(next_z, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * log_pi
        target_q = reward + (1.0 - done) * self.gamma * next_q
        loss_critic = (curr_q1 - target_q).pow_(2).mean() + (curr_q2 - target_q).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        self.optim_critic.step()

        if self.learning_steps_sac % 1000 == 0:
            # print(f"loss_critic:{loss_critic:.4f}")
            writer.add_scalar("loss/critic", loss_critic.item(), self.learning_steps_sac)

    def update_actor(self, z, feature_action, writer):
        action, log_pi = self.actor.sample(feature_action)
        q1, q2 = self.critic(z, action)
        loss_actor = -torch.mean(torch.min(q1, q2) - self.alpha * log_pi)

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

        with torch.no_grad():
            entropy = -log_pi.detach().mean()
        loss_alpha = -self.log_alpha * (self.target_entropy - entropy)

        self.optim_alpha.zero_grad()
        loss_alpha.backward(retain_graph=False)
        self.optim_alpha.step()
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()

        if self.learning_steps_sac % 1000 == 0:
            writer.add_scalar("loss/actor", loss_actor.item(), self.learning_steps_sac)
            writer.add_scalar("loss/alpha", loss_alpha.item(), self.learning_steps_sac)
            writer.add_scalar("stats/alpha", self.alpha.item(), self.learning_steps_sac)
            writer.add_scalar("stats/entropy", entropy.item(), self.learning_steps_sac)

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # We don't save target network to reduce workloads.
        torch.save(self.latent.encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
        torch.save(self.latent.state_dict(), os.path.join(save_dir, "latent.pth"))
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))
