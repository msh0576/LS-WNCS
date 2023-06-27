import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import collections
import numpy as np
import random
import math
import os

from Envs.env import delay_step, get_delay, dt_step
from Models.functions import scheduler
from Models.running_stats import RunningStats, preprocess_norm
from Models.delay_sac.buffer import ReplayBuffer as ReplayBuffer_v2
from Models.delay_sac.utils import create_feature_actions
from Models.delay_sac.utils import reparameterize


# import mbrl_smdp_ode_master.utils as utils

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

def to_hybrid_action(action_c, action_d, flat_actions=True):
    '''
    Input:
        action_c: numpy, 
        action_d: int
    Output:
        action: list, [action_d (=scalar), action_c (=D_action_c)]
    '''
    ac = action_c.tolist()
    ad = [float(action_d)]
    return ad + ac



def to_torch_action(actions, device):
    ad = torch.Tensor(actions[:, 0]).int().to(device)
    ac = torch.Tensor(actions[:, 1:]).to(device)
    return ac, ad

class Policy(nn.Module):
    def __init__(self, input_shape, out_c, out_d, device):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.mean = nn.Linear(64, out_c)
        self.logstd = nn.Linear(64, out_c)
        self.pi_d = nn.Linear(64, out_d)
        self.apply(layer_init)

        self.device = device

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.Tensor(x).to(self.device)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = F.leaky_relu(self.fc3(x), 0.01)
        mean = torch.tanh(self.mean(x))
        log_std = torch.tanh(self.logstd(x))
        pi_d = self.pi_d(x)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, pi_d

    def get_action(self, x):
        mean, log_std, pi_d = self.forward(x)
        # std = log_std.exp()
        # normal = Normal(mean, std)
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # action_c = torch.tanh(x_t)
        # log_prob_c = normal.log_prob(x_t)
        # log_prob_c -= torch.log(1.0 - action_c.pow(2) + 1e-8)

        action_c,log_prob_c = reparameterize(mean, log_std)


        dist = Categorical(logits=pi_d)
        action_d = dist.sample()    # where +1 means range of categorical [0, max] -> [1, max+1]
        prob_d = dist.probs
        log_prob_d = torch.log(prob_d + 1e-8)

        return action_c, action_d, log_prob_c, log_prob_d, prob_d

    def to(self, device):
        return super(Policy, self).to(device)

class SoftQNetwork(nn.Module):
    def __init__(self, input_shape, out_c, out_d, layer_init, device):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape + out_c, 256)
        # self.fc2 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, out_d)
        self.apply(layer_init)

        self.device = device

    def forward(self, x, a):
        if type(x) == np.ndarray:
            x = torch.Tensor(x).to(self.device)
        x = torch.cat([x, a], 1)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = F.leaky_relu(self.fc3(x), 0.01)
        # x = F.leaky_relu(self.fc4(x), 0.01)
        x = self.fc4(x)
        return x

class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)

class ReplayBuffer_Trajectory():
    def __init__(self, size, observation_size, action_c_size, action_d_size):
        # self.buffer = collections.deque(maxlen=buffer_limit)
        self.size = size
        self.observations = np.empty((size, observation_size), dtype=np.float32)
        self.actions_c = np.empty((size, action_c_size), dtype=np.float32)
        self.actions_d = np.empty((size, 1), dtype=np.float32)
        self.rewards = np.empty((size,), dtype=np.float32)
        self.nonterminals = np.empty((size,1), dtype=np.float32)
        self.idx = 0
        self.full = False
        self.steps, self.episodes = 0, 0


    def append(self, observation, action_c, action_d, reward, done):
        '''
        Input:
            observation: np, []
            action_c: np,
            action_d: np,
            reward: scalar
            done: bool
        '''
        if type(observation) == torch.tensor:
            observation = observation.detach().cpu().numpy()

        self.observations[self.idx] = observation
        self.actions_c[self.idx] = action_c
        self.actions_d[self.idx] = action_d
        self.rewards[self.idx] = reward
        self.nonterminals[self.idx] = not done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)
    
    
    
    def sample(self, n):
        '''
        Output:
            observations: np, [n, D_state]
            actions_c: np, [n, D_action_c]
            actions_d: np, [n, D_action_d]
            rewrds: np, [n, ]
            nonterminals: np, [n, 1]
        '''
        idxs = np.random.choice(self.size if self.full else self.idx, n)
        obs_list, action_c_list, action_d_list, reward_list, nonterminal_list = [], [], [] ,[], []
        for idx in idxs:
            obs_list.append(self.observations[idx])
            action_c_list.append(self.actions_c[idx])
            action_d_list.append(self.actions_d[idx])
            reward_list.append(self.rewards[idx])
            nonterminal_list.append(self.nonterminals[idx])
        
        return np.array(obs_list), np.array(action_c_list), np.array(action_d_list), np.array(reward_list), np.array(nonterminal_list)

    # Returns an index for a valid single sequence chunk uniformly sampled from the memory
    def _sample_idx(self, L):
        '''
        <Return>
            idxs: random idx를 뽑고, idx ~ idx+L 까지의 정수 array
        '''
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
        return idxs
    
    def _retrieve_batch(self, idxs, n, L):
        '''
        <Return>
            self.observations, actions, rewards, ... 에서 vec_idxs 의 sample을 뽑아내고, [L, n, dim] 으로 reshape해서 return
            즉, batch 개수 (=n)만큼 하나의 열에 나열하고, 이를 trajectory (=L) 수 만큼 행으로 나열된 형태 : 하나의 열이 하나의 trajectory 정보를 가짐
            # ex) n=2, L=4,
            #     idxs = [[1, 2, 3, 4]
            #             [5, 6, 7, 8]]
            #     then,
            #     vec_idxs = [1, 5, 2, 6, 3, 7, 4, 8]
        '''
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices

        observations = self.observations[vec_idxs].reshape(L, n, -1)
        actions_c = self.actions_c[vec_idxs].reshape(L, n, -1)
        actions_d = self.actions_d[vec_idxs].reshape(L, n, -1)
        rewards = self.rewards[vec_idxs].reshape(L, n)
        nonterminals = self.nonterminals[vec_idxs].reshape(L, n, 1)
        # print("observations shape:", observations.shape)
        # print("actions_c:", actions_c.shape)
        # print("actions_d:", actions_d.shape)
        
        return observations, actions_c, actions_d, rewards, nonterminals
    
    # Returns a batch of sequence chunks uniformly sampled from the memory
    def sample_traj(self, batch, L):
        '''
        <Argument>
            batch: batch size
            L: chunk size: 하나의 batch의 trajectory length
        '''
        batch = self._retrieve_batch(np.asarray([self._sample_idx(L) for _ in range(batch)]), batch, L)
        
        return [item for item in batch] # list, [observations, actions_c, action_d, rewards, nonterminals]

    def __len__(self):
        return self.idx if not self.full else self.size

class HsacAlgorithm:
    def __init__(
        self,
        buffer_size, 
        batch_size, 
        input_shape, 
        out_c, 
        out_d, 
        device,
        num_plant=7, 
        q_lr=1e-3, 
        policy_lr=1e-4, 
        autotune=True, 
        ent_c=-0.25, 
        ent_d=-0.25,
        alpha=0.2, 
        policy_frequency=1, 
        target_network_frequency=1, 
        tau=0.1, 
        gamma=0.9,
        obs_normal=True,
        num_sequences=8,
        sf_sched=False,
    ):
        self.batch_size = batch_size
        self.device = device
        self.autotune = autotune
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau
        self.gamma = gamma
        self.schedule_size = out_d
        self.num_plant = num_plant
        self.num_sequences = num_sequences
        self.min_period = 5
        self.max_period = out_d - 1
        self.sf_len = out_d - 1
        self.sf_sched = sf_sched
        self.use_hybrid_action = None
        self.exploration_check = False # False: default | True: 초기 샘플 수집 후 policy training 없이 state and action exploration (distribution)이 어떻게 나오는지 확인하기 위해

        if sf_sched:
            self.hybrid_action_dim = out_c*math.ceil(self.sf_len/self.min_period)
        else:
            self.hybrid_action_dim = out_c


        print(f"num_sequences:{num_sequences}")
        # running mean and std of observations
        self.rms = RunningStats(dim=input_shape, device=device) if obs_normal else None

        self.rb = ReplayBuffer(buffer_size)
        self.rb_traj = ReplayBuffer_Trajectory(buffer_size, observation_size=input_shape, action_c_size=out_c, action_d_size=out_d)
        self.buffer = ReplayBuffer_v2(buffer_size, num_sequences, (input_shape,), (2,), device, use_image=False, rms=self.rms)

        # self.pg = Policy(input_shape*num_sequences + out_c*(num_sequences-1), out_c, out_d, device).to(device)
        # self.qf1 = SoftQNetwork(input_shape*num_sequences, out_c, out_d, layer_init, device).to(device)
        # self.qf2 = SoftQNetwork(input_shape*num_sequences, out_c, out_d, layer_init, device).to(device)
        # self.qf1_target = SoftQNetwork(input_shape*num_sequences, out_c, out_d, layer_init, device).to(device)
        # self.qf2_target = SoftQNetwork(input_shape*num_sequences, out_c, out_d, layer_init, device).to(device)
        
        # === vanillar sac ===
        self.pg = Policy(input_shape, self.hybrid_action_dim, out_d, device).to(device)
        self.qf1 = SoftQNetwork(input_shape, self.hybrid_action_dim, out_d, layer_init, device).to(device)
        self.qf2 = SoftQNetwork(input_shape, self.hybrid_action_dim, out_d, layer_init, device).to(device)
        self.qf1_target = SoftQNetwork(input_shape, self.hybrid_action_dim, out_d, layer_init, device).to(device)
        self.qf2_target = SoftQNetwork(input_shape, self.hybrid_action_dim, out_d, layer_init, device).to(device)

        # === superframe sac ===
        # print(f"hybrid_action_dim:{self.hybrid_action_dim}")

        # self.pg = Policy(input_shape, self.hybrid_action_dim, out_d, device).to(device)
        # self.qf1 = SoftQNetwork(input_shape, self.hybrid_action_dim, out_d, layer_init, device).to(device)
        # self.qf2 = SoftQNetwork(input_shape, self.hybrid_action_dim, out_d, layer_init, device).to(device)
        # self.qf1_target = SoftQNetwork(input_shape, self.hybrid_action_dim, out_d, layer_init, device).to(device)
        # self.qf2_target = SoftQNetwork(input_shape, self.hybrid_action_dim, out_d, layer_init, device).to(device)

        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.values_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr)
        self.policy_optimizer = optim.Adam(list(self.pg.parameters()), lr=policy_lr)
        self.loss_fn = nn.MSELoss()

        # network scheduler
        self.random_scheduler = scheduler(algo='random', schedule_len=num_plant+1, max_period=self.schedule_size-1)

        # JIT compile to speed up.
        fake_feature = torch.empty(1, num_sequences + 1, input_shape, device=device)
        fake_action = torch.empty(1, num_sequences, self.hybrid_action_dim, device=device)
        self.create_feature_actions = torch.jit.trace(create_feature_actions, (fake_feature, fake_action))

        # Automatic entropy tuning
        if autotune:
            # target_entropy = -float(out_c)
            self.target_entropy = ent_c
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().detach().cpu().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=1e-4)

            # target_entropy_d = -0.98 * np.log(1/out_d)
            self.target_entropy_d = ent_d
            self.log_alpha_d = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_d = self.log_alpha_d.exp().detach().cpu().item()
            self.a_d_optimizer = optim.Adam([self.log_alpha_d], lr=1e-4)
        else:
            self.alpha = alpha
            self.alpha_d = alpha
        
        # running mean and std of observations
        self.rms = RunningStats(dim=input_shape, device=device) if obs_normal else None

        # Other parameters
        self.prev_state = None
        self.prev_action = None

    def preprocess(self, ob):
        if self.rms is not None:
            state = self.rms.unnormalize(ob.state).view(1, -1)
        else:
            state = torch.tensor(ob.state, dtype=torch.float, device=self.device).view(1, -1)   # [1, T*D_state]
        action = torch.tensor(ob.action, dtype=torch.float, device=self.device).view(1, self.num_sequences-1, -1) # [1, T-1, 2]     
        action_c = action[:, :, 1] # [1, T-1]
        action_d = action[:, :, 0] # [1, T-1]
        state_action = torch.cat([state, action_c], dim=-1) # [1, T*D_state + (T-1)*1]
        return state_action

    # def explore(self, ob):
    #     state_action = self.preprocess(ob)
    #     with torch.no_grad():
    #         action_c, action_d, _, _, _ = self.pg.get_action(state_action) # [1, 1] | [1,]
    #         # print(f"action_c:{action_c.shape} | action_d:{action_d.shape}")
    #     return action_c.squeeze(0).cpu().numpy(), action_d.cpu().numpy()
    
    def explore(self, ob):
        # if type(ob) != torch.tensor:
        #     ob = torch.tensor(ob, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_c, action_d, _, _, _ = self.pg.get_action(ob) # [1, 1] | [1,]
            # print(f"action_c:{action_c.shape} | action_d:{action_d.shape}")
        return action_c.cpu().numpy(), action_d.cpu().numpy()

    def exploit(self, ob):
        feature_action = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor(feature_action)
            # action = self.actor(feature_action)[0]
        return action.cpu().numpy()[0]

    def random_act(self, env):
        action_c = env.action_space.sample()
        # action_d = int(self.random_scheduler.get_period(syst_idx=0))
        action_d = random.randint(
            self.min_period, self.max_period, 
            # size=(1,)
        )
        dt = self.process_daction(action_d)
        action = to_hybrid_action(action_c, action_d)   # [action_d, [action_c]]
        return action_c, action_d, dt, action
        
        

    def step(self, env, ob, t, is_random, delay_step_=False):
        if self.prev_action == None:
            self.prev_action = env.action_space.sample()
        t += 1
        
        # == select action ==
        if is_random:
            # == select schedule ==
            # action_d = int(self.random_scheduler.get_period(syst_idx=0))
            # dt = self.process_daction(action_d)
            # action_c = env.action_space.sample()
            # action = to_hybrid_action(action_c, action_d)   # [action_d, [action_c]]
            action_c, action_d, dt, action = self.random_act(env)
        else:
            # action_c, action_d = self.explore(ob)   # [1,] | [1,]
            action_c, action_d = self.explore(self.prev_state)   # [1,] | [1,]
            dt = self.process_daction(action_d)
            action = to_hybrid_action(action_c, action_d)   # [action_d, action_c]
        # print(f"action_c:{action_c} | action_d:{action_d} | action:{action} | dt:{dt}")


        state, reward, done, _ = dt_step(env, action_c, dt=dt)

        mask = False if t == env.spec.max_episode_steps else done
        self.memory_push(self.prev_state, action, reward, state, done)

        ob.append(state, action)
        # self.buffer.append(action, reward, mask, state, done)
        self.memory_push_traj(state, action_c, action_d, reward, done)


        self.prev_action = action
        self.prev_state = state

        # == state normalization prepare ==
        if self.rms is not None:
            self.rms += state 

        if done:
            t = 0
            state = env.reset()
            self.prev_action = None
            self.prev_state = state
            ob.reset_episode(state)
            # self.buffer.reset_episode(state)

            if self.rms is not None:
                self.rms += state 

        return t, reward, done
    
    def evaluate_steps(self, env, delay_step_=False):
        state = env.reset()
        episode_return = 0.0
        done = False
        prev_state = state
        dt_ratio = [0 for _ in range(self.schedule_size)]
        cont_cunt = 0
        trajectory = {
            'state':[],
            'action':[],
            'dt':[],
            'latent_action':[],
            'hybrid_action':[],
        }
        
        while not done:
            cont_cunt += 1
            action_c, action_d = self.explore(prev_state)
            dt = self.process_daction(action_d)
            action = to_hybrid_action(action_c, action_d)   # [action_d, action_c]
            state, reward, done, _ = dt_step(env, action_c, dt)
            prev_state = state
            episode_return += reward
            dt_ratio[dt-1] += 1
            trajectory['state'].append(state)
            trajectory['hybrid_action'].append(action)

        trajectory['state'] = np.stack(trajectory['state'], axis=0)
        trajectory['hybrid_action'] = np.stack(trajectory['hybrid_action']).squeeze()
        
        return episode_return, dt_ratio, trajectory, cont_cunt

    def superframe(self):
        dt = np.random.randint(self.min_period, self.max_period+1)

        dts = [dt for _ in range(self.sf_len//dt)]
        if self.sf_len - sum(dts) > 0:
            dts.append(self.sf_len - sum(dts))
        # print(f"dt:{dt} | dts:{dts}")
        return dts

    def sf_act(self, obs):
        with torch.no_grad():
            action_c, action_d, _, _, _ = self.pg.get_action(obs) # [D_caction] | scalar
        k = max(action_d.item(), self.min_period)
        # === schedule in a SF ===
        dts = [k] * (self.sf_len // k )    
        dt_resid = self.sf_len - sum(dts)
        if dt_resid > 0:
            dts.append(dt_resid)

        # === make action : [action_c, k_norm] ===
        k_norm = self.k_normal(k)
        assert k_norm <= 1. and k_norm >= -1., f"k:{k}"
        action = torch.cat([torch.tensor(k_norm, dtype=torch.float), action_c.cpu()])
        return action.numpy(), dts, action_c.cpu().numpy()
        

    def sf_step(self, env, ob, t, is_random, delay_step_=False):
        sf_reward = 0.
        action_vec_list = []

        # == superframe period ==
        dts = self.superframe()

        if not is_random:
            action_vec, dts, cactions = self.sf_act(self.prev_state)
            # print(f"cactions:{cactions} | dts:{dts}")
        for idx, dt in enumerate(dts):
            # == select continuous action ==
            if is_random:
                caction = env.action_space.sample()
                if len(action_vec_list) < self.hybrid_action_dim - 1:
                    action_vec_list.append(caction)
            else:
                caction = cactions[idx:idx+1]
            # == step env ==
            state, reward, done, _ = dt_step(env, caction, dt=dt)
            sf_reward += reward
            t += dt
            if done:
                t = 0
                state = env.reset()
            
        if is_random:
            action_vec_numpy = np.array(action_vec_list).reshape(-1)
            k = self.k_normal(dts[0])
            action_vec = np.concatenate([k, self.pad_action(action_vec_numpy)])
        else:
            # if generated action_vec elements are not used, it is replaced to 0
            if idx < len(action_vec) - 1:
                action_vec[idx+2:] = 0.
        # print(f"dts:{dts} | action_vec;{action_vec}")
        
        # == memorize transition ==
        self.memory_push(self.prev_state, action_vec, sf_reward, state, done)
        # == env postprocess ==
        self.prev_state = state
        

        return t, reward, done


    def evaluate_sf_steps(self, env):
        state = env.reset()
        prev_state = state
        episode_return = 0.0
        done = False
        dt_ratio = [0 for _ in range(self.sf_len+1)]

        while not done:
            
            action_vec, dts, cactions = self.sf_act(prev_state)
            # print(f"dts:{dts} | action_vec:{action_vec}")
            for idx, dt in enumerate(dts):
                # == select continuous action ==
                action = cactions[idx:idx+1]
                
                state, reward, done, _ = dt_step(env, action, dt)
                episode_return += reward

                if done:
                    break
            dt_ratio[dts[0]-1] += 1
            prev_state = state

        return episode_return, dt_ratio


    def train_tmp_v2(self, global_step):
        if len(self.buffer) > self.batch_size:
            # state_: [B, T+1, D_state] | action_: [B, T, 2] | reward: [B,] | done: [B,]
            state_, action_, reward, done = self.buffer.sample_sac(self.batch_size)
            # print(f"state_:{state_.shape} | action_:{action_.shape} | reward:{reward.shape} | done:{done.shape}")

            curr_observations = state_[:, :-1, :].view(self.batch_size, -1)    # [B, T*D_state]
            next_observations = state_[:, 1:, :].view(self.batch_size, -1)      # [B, T*D_state]
            rewards = reward.view(-1)               # [B,]
            dones = done.view(-1)              # [B,]
            actions_c = action_[:, :, 1:]    # [B, T, 1]
            actions_d = action_[:, :, :1]    # [B, T, 1]
            action_c = action_[:, -1, 1:]
            action_d = action_[:, -1, :1]

            # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t))
            feature_action, next_feature_action = self.create_feature_actions(state_, actions_c)  # [B, (T*D_state) + ((T-1)*D_action_c)]
            with torch.no_grad():
                next_state_actions_c, next_state_actions_d, next_state_log_pi_c, next_state_log_pi_d, next_state_prob_d = self.pg.get_action(next_feature_action)
                # next_state_actions_c: [B, 1]
                # print(f"next_state_actions_c:{next_state_actions_c.shape}")
                qf1_next_target = self.qf1_target.forward(next_observations, next_state_actions_c) # [B, D_action_d]
                qf2_next_target = self.qf2_target.forward(next_observations, next_state_actions_c)
                
                min_qf_next_target = next_state_prob_d * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_prob_d * next_state_log_pi_c - self.alpha_d * next_state_log_pi_d)  # [B, 15 (=D_action_d)]

                # === get return ===
                next_q_value = rewards + (1 - dones) * self.gamma * (min_qf_next_target.sum(1)).view(-1)    # [B, ]

            # s_actions_c, s_actions_d = to_torch_action(s_actions, self.device)
            qf1_a_values = self.qf1.forward(curr_observations, action_c).gather(1, action_d.long().view(-1, 1).to(self.device)).squeeze().view(-1)
            qf2_a_values = self.qf2.forward(curr_observations, action_c).gather(1, action_d.long().view(-1, 1).to(self.device)).squeeze().view(-1)

            qf1_loss = self.loss_fn(qf1_a_values, next_q_value)
            qf2_loss = self.loss_fn(qf2_a_values, next_q_value)
            qf_loss = (qf1_loss + qf2_loss) / 2
            
            self.values_optimizer.zero_grad()
            qf_loss.backward()
            self.values_optimizer.step()

            if global_step % self.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(self.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    actions_c, actions_d, log_pi_c, log_pi_d, prob_d = self.pg.get_action(feature_action)
                    qf1_pi = self.qf1.forward(curr_observations, actions_c)
                    qf2_pi = self.qf2.forward(curr_observations, actions_c)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)

                    policy_loss_d = (prob_d * (self.alpha_d * log_pi_d - min_qf_pi)).sum(1).mean()
                    policy_loss_c = (prob_d * (self.alpha * prob_d * log_pi_c - min_qf_pi)).sum(1).mean()
                    policy_loss = policy_loss_d + policy_loss_c

                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()

            # update the target network
            if global_step % self.target_network_frequency == 0:
                for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def n_step_return(self, rewards, final_value, dts, gamma=0.9):
        '''
        Input:
            rewards: [L, B,]
            dts: [T, B, 1]
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
        output_ += (gamma ** N) * final_value
        return output_

    def train(self, global_step, writer):
        if len(self.rb.buffer) > self.batch_size:  # starts update as soon as there is enough data.
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = self.rb.sample(self.batch_size)
            
            # === for sf_step ===
            # s_obs = torch.tensor(s_obs, dtype=torch.float32, device=self.device)
            # s_actions = torch.tensor(s_actions, dtype=torch.float32, device=self.device)
            # s_rewards = torch.tensor(s_rewards, dtype=torch.float32, device=self.device)
            # s_next_obses = torch.tensor(s_next_obses, dtype=torch.float32, device=self.device)
            # s_dones = torch.tensor(s_dones, dtype=torch.float32, device=self.device)
        
            with torch.no_grad():
                next_state_actions_c, next_state_actions_d, next_state_log_pi_c, next_state_log_pi_d, next_state_prob_d = self.pg.get_action(s_next_obses)
                qf1_next_target = self.qf1_target.forward(s_next_obses, next_state_actions_c)
                qf2_next_target = self.qf2_target.forward(s_next_obses, next_state_actions_c)

                # print("qf1_next_target:{} | next_state_prob_d:{} | next_state_log_pi_c:{} | next_state_log_pi_d:{}".format(qf1_next_target.shape, next_state_prob_d.shape, next_state_log_pi_c.shape, next_state_log_pi_d.shape))
                min_qf_next_target = next_state_prob_d * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_prob_d * next_state_log_pi_c - self.alpha_d * next_state_log_pi_d)  # [B, D_action_d]
                # === for sf_step ===
                # next_q_value = s_rewards + (1 - s_dones) * self.gamma * (min_qf_next_target.sum(1)).view(-1)
                next_q_value = torch.Tensor(s_rewards).to(self.device) + (1 - torch.Tensor(s_dones).to(self.device)) * (self.gamma ** next_state_actions_d) * (min_qf_next_target.sum(1)).view(-1)
            
            if self.sf_sched:
                # === for sf_step ===
                s_actions_c = torch.tensor(s_actions[:, 1:], dtype=torch.float, device=self.device)
                s_actions_d = torch.tensor(list(map(lambda x:self.k_unnormal(x), s_actions[:, 0])), dtype=torch.int, device=self.device)
            else:
                # === for slot_step ===
                s_actions_c, s_actions_d = to_torch_action(s_actions, self.device)
            # print(f"c:{s_actions_c.shape} | d:{s_actions_d.shape}")

            qf1_a_values = self.qf1.forward(s_obs, s_actions_c).gather(1, s_actions_d.long().view(-1, 1).to(self.device)).squeeze().view(-1)
            qf2_a_values = self.qf2.forward(s_obs, s_actions_c).gather(1, s_actions_d.long().view(-1, 1).to(self.device)).squeeze().view(-1)
            qf1_loss = self.loss_fn(qf1_a_values, next_q_value)
            qf2_loss = self.loss_fn(qf2_a_values, next_q_value)
            qf_loss = (qf1_loss + qf2_loss) / 2

            self.values_optimizer.zero_grad()
            qf_loss.backward()
            self.values_optimizer.step()

            # if global_step % self.policy_frequency == 0:  # TD 3 Delayed update support
            for _ in range(
                    self.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                actions_c, actions_d, log_pi_c, log_pi_d, prob_d = self.pg.get_action(s_obs)
                qf1_pi = self.qf1.forward(s_obs, actions_c)
                qf2_pi = self.qf2.forward(s_obs, actions_c)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)

                policy_loss_d = (prob_d * (self.alpha_d * log_pi_d - min_qf_pi)).sum(1).mean()
                policy_loss_c = (prob_d * (self.alpha * prob_d * log_pi_c - min_qf_pi)).sum(1).mean()
                # print("policy_loss_c:", policy_loss_c)
                policy_loss = policy_loss_d + policy_loss_c

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                if self.autotune:
                    with torch.no_grad():
                        a_c, a_d, lpi_c, lpi_d, p_d = self.pg.get_action(s_obs)
                    alpha_loss = (-self.log_alpha * p_d * (p_d * lpi_c + self.target_entropy)).sum(1).mean()
                    alpha_d_loss = (-self.log_alpha_d * p_d * (lpi_d + self.target_entropy_d)).sum(1).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().detach().cpu().item()

                    self.a_d_optimizer.zero_grad()
                    alpha_d_loss.backward()
                    self.a_d_optimizer.step()
                    self.alpha_d = self.log_alpha_d.exp().detach().cpu().item()

            # update the target network
            if global_step % self.target_network_frequency == 0:
                for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def memory_push_traj(self, obs, action_c, action_d, reward, done):
        self.rb_traj.append(obs, action_c, action_d, reward, done)
    
    # train with trajectory
    def train_traj(self, global_step):
        if len(self.rb_traj) > self.batch_size:  # starts update as soon as there is enough data.
            observations, actions_c, actions_d, rewards, nonterminals = self.rb_traj.sample_traj(self.batch_size, L=10)

            curr_observations = torch.tensor(observations, dtype=torch.float32, device=self.device)[:-1]    # [L-1, B, D_state]
            next_observations = torch.tensor(observations, dtype=torch.float32, device=self.device)[1:]
            actions_c = torch.tensor(actions_c, dtype=torch.float32, device=self.device)[:-1]
            actions_d = torch.tensor(actions_d, dtype=torch.float32, device=self.device)[:-1]
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)[:-1]               # [L-1, B, ]
            nonterminals = torch.tensor(nonterminals, dtype=torch.float32, device=self.device)[:-1].squeeze(1)     # [L-1, B, ]

            # print("curr_obs shape:", curr_observations.shape)
            with torch.no_grad():
                next_state_actions_c, next_state_actions_d, next_state_log_pi_c, next_state_log_pi_d, next_state_prob_d = self.pg.get_action(next_observations[-1])
                qf1_next_target = self.qf1_target.forward(next_observations[-1], next_state_actions_c) # [B, D_action_d]
                qf2_next_target = self.qf2_target.forward(next_observations[-1], next_state_actions_c)
                
                min_qf_next_target = next_state_prob_d * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_prob_d * next_state_log_pi_c - self.alpha_d * next_state_log_pi_d)
                # === get return ===
                next_q_value = self.n_step_return(rewards, min_qf_next_target.sum(1), actions_d.squeeze(-1))
                # next_q_value = torch.Tensor(s_rewards).to(self.device) + (1 - torch.Tensor(s_dones).to(self.device)) * self.gamma * (min_qf_next_target.sum(1)).view(-1)
            
            # s_actions_c, s_actions_d = to_torch_action(s_actions, self.device)
            qf1_a_values = self.qf1.forward(curr_observations[0], actions_c[0]).gather(1, actions_d[0].long().view(-1, 1).to(self.device)).squeeze().view(-1)
            qf2_a_values = self.qf2.forward(curr_observations[0], actions_c[0]).gather(1, actions_d[0].long().view(-1, 1).to(self.device)).squeeze().view(-1)

            qf1_loss = self.loss_fn(qf1_a_values, next_q_value)
            qf2_loss = self.loss_fn(qf2_a_values, next_q_value)
            qf_loss = (qf1_loss + qf2_loss) / 2
            
            self.values_optimizer.zero_grad()
            qf_loss.backward()
            self.values_optimizer.step()

            if global_step % self.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(self.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    actions_c, actions_d, log_pi_c, log_pi_d, prob_d = self.pg.get_action(curr_observations[0])
                    qf1_pi = self.qf1.forward(curr_observations[0], actions_c)
                    qf2_pi = self.qf2.forward(curr_observations[0], actions_c)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)

                    policy_loss_d = (prob_d * (self.alpha_d * log_pi_d - min_qf_pi)).sum(1).mean()
                    policy_loss_c = (prob_d * (self.alpha * prob_d * log_pi_c - min_qf_pi)).sum(1).mean()
                    policy_loss = policy_loss_d + policy_loss_c

                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()

            # update the target network
            if global_step % self.target_network_frequency == 0:
                for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    

    def memory_push(self, obs, action, reward, next_obs, done):
        self.rb.put((obs, action, reward, next_obs, done))
    
    def process_daction(self, daction):
        '''
        daction: discrete action has a boundary [0 ~ D_daction]
                daction means a transmission period (a time interval to receive a next state)
        Output:
            schedule: scalar [1 ~ D_daction+1]
        '''
        return daction+1
    
    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save(self.pg.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.qf1.state_dict(), os.path.join(path, "critic1.pth"))
        torch.save(self.qf2.state_dict(), os.path.join(path, "critic2.pth"))
    
    def load_model(self, path):
        self.pg.load_state_dict(os.path.join(path, "actor.pth"))
        self.qf1.load_state_dict(os.path.join(path, "critic1.pth"))
        self.qf2.load_state_dict(os.path.join(path, "critic2.pth"))
    
    def pad_action(self, action_vec):
        if len(action_vec) == self.hybrid_action_dim:
            return action_vec
        pad = np.zeros(self.hybrid_action_dim - len(action_vec), dtype=np.float)
        pad_action_vec = np.concatenate([action_vec, pad])
        return pad_action_vec
        
    def k_unnormal(self, k_norm):
        # return round(k_norm*(self.max_period - self.min_period) + self.min_period)
        return round( (k_norm+1) / 2 * (self.max_period - self.min_period) + self.min_period)
    
    def k_normal(self, k):
        # return np.array([(k - self.min_period) / (self.max_period - self.min_period)], dtype=np.float)
        return np.array([2 * (k - self.min_period) / (self.max_period - self.min_period) - 1], dtype=np.float)