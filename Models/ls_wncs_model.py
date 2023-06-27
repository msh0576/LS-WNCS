import torch
import numpy as np
import math
from copy import deepcopy
from collections import Counter
import random

from Models.latent_model import Latent
from Models.vae_model import VAEModule
from Models.functions import scheduler
from Models.util_model import k_to_dts, dts_to_k,\
    select_syst, schedulable_check, epsilon_decay, \
    get_curr_time, timer_check
from Envs.env import dt_step


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(2e6)):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.storage = dict()
        self.storage['state'] = np.zeros((max_size, state_dim))
        self.storage['action'] = np.zeros((max_size, action_dim))
        self.storage['next_state'] = np.zeros((max_size, state_dim))
        self.storage['reward'] = np.zeros((max_size, 1))
        self.storage['not_done'] = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.storage['state'][self.ptr] = state
        self.storage['action'][self.ptr] = action
        self.storage['next_state'][self.ptr] = next_state
        self.storage['reward'][self.ptr] = reward
        self.storage['not_done'][self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.storage['state'][ind]).to(self.device),
            torch.FloatTensor(self.storage['action'][ind]).to(self.device),
            torch.FloatTensor(self.storage['next_state'][ind]).to(self.device),
            torch.FloatTensor(self.storage['reward'][ind]).to(self.device),
            torch.FloatTensor(self.storage['not_done'][ind]).to(self.device)
        )
    
    # Returns an index for a valid single sequence chunk uniformly sampled from the memory
    def _sample_idx(self, L):
        '''
        <Return>
            idxs: random idx를 뽑고, idx ~ idx+L 까지의 정수 array
        '''
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.max_size if self.size==self.max_size else self.ptr - L)
            idxs = np.arange(idx, idx + L) % self.size
            valid_idx = not self.ptr in idxs[1:]  # Make sure data does not cross the memory index
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

        states = self.storage['state'][vec_idxs].reshape(L, n, -1)
        actions = self.storage['action'][vec_idxs].reshape(L, n, -1)
        next_states = self.storage['next_state'][vec_idxs].reshape(L, n, -1)
        rewards = self.storage['reward'][vec_idxs].reshape(L, n)
        not_dones = self.storage['not_done'][vec_idxs].reshape(L, n, 1)
        # print("observations shape:", observations.shape)
        # print("actions_c:", actions_c.shape)
        # print("actions_d:", actions_d.shape)
        
        return (
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(not_dones).to(self.device)
        )
    
    # Returns a batch of sequence chunks uniformly sampled from the memory
    def sample_traj(self, batch, L):
        '''
        <Argument>
            batch: batch size
            L: chunk size: 하나의 batch의 trajectory length
        '''
        batch = self._retrieve_batch(np.asarray([self._sample_idx(L) for _ in range(batch)]), batch, L)
        
        return [item for item in batch] # list, [states, actions, next_states, rewards, nonterminals]

    def __len__(self):
        return self.size

    def save(self, filename):
        np.save("./buffers/" + filename + ".npy", self.storage)

    def load(self, data):
        assert('next_observations' in data.keys())
        for i in range(data['observations'].shape[0] - 1):
            self.add(data['observations'][i], data['actions'][i], data['next_observations'][i],
                     data['rewards'][i], data['terminals'][i])
        print("Dataset size:" + str(self.size))



class LSWNCSAlgorithm():
    def __init__(
        self,
        state_shape,
        action_shape,
        schedule_size,
        max_period,
        min_period,
        sf_len,
        latent_dim,
        device,
        batch_size,
        algo_name,
        sf_sched=True,
        num_plant=7,
        vae_lr=1e-4,
        vae_hidden_size=750,
        nstep_return=False,
        multipolicy=False,
        **kwargs,
    ):  
        self.state_shape = state_shape.shape[0]
        self.action_dim = action_shape.shape[0]
        self.max_action = float(action_shape.high)
        self.latent_dim = latent_dim
        self.algo_name = algo_name
        self.device= device
        self.kwargs = kwargs
        self.sf_len = sf_len
        self.max_period = max_period    # 30: basic |
        self.min_period = min_period # 5: basic |
        num_sequences = kwargs['num_sequences']
        obs_normal = kwargs['obs_normal']
        self.policy_name = 'sac'
        self.sf_sched = sf_sched
        self.num_plant = num_plant
        self.nstep_return = nstep_return
        self.dec_explo = True
        dep_check = True
        loss_dyn = True
        loss_energy = kwargs['loss_energy']
        self.energy_coeff = kwargs['energy_coeff']
        self.dtdiscount = 0.99  # 0.99: reward discounted by discrete action | 1.: no discount
        self.all_reward = False
        self.future_sched = 0  # 0: not use | max_period
        self.eps_decay = None   # 1e-5: recommand | 1e-6*5 | 0: static | None: not use 
        self.action_relabel = True
        self.multi_policy = multipolicy
        self.eval_random_act = None # 'origin': exploration on original space (original action을 랜덤하게 선택 + no policy training) | 'latent': exploration on latent space (z를 랜덤하게 선택 + no policy training) | None: basic (evaluation 때 학습한 모델 사용)
        self.novae = False  # False: default | vae training 없이 latent action을 action으로 사용했을 때, random exploration 으로 학습 가능한지 확인하기 위해
        self.exploration_check = False # False: default | True: vae training 후 policy training 없이 state and action exploration (distribution)이 어떻게 나오는지 확인하기 위해

        print(f"dec_explo:{self.dec_explo} | dep_check:{dep_check} | dyn_loss:{loss_dyn} | nstep:{nstep_return} | ",
            f"decay_eps:{self.eps_decay} | action_relabel:{self.action_relabel} |",
            f"multi_policy:{self.multi_policy} | dtdiscount:{self.dtdiscount} |",
            f"eval_random_act:{self.eval_random_act} | novae:{self.novae} |",
            f"exploration_check:{self.exploration_check} |")
        
        if sf_sched:    # daction = dts
            self.state_dim = (state_shape.shape[0] * num_plant) + (1 * num_plant)
            self.cont_action_dim = self.action_dim * math.ceil(sf_len/self.min_period) * num_plant
            self.disc_action_dim = math.ceil(sf_len/self.min_period) * num_plant
        else:   # daction = syst_id at each slot
            if self.future_sched > 0:
                self.state_dim = (state_shape.shape[0]) + self.future_sched
            else:
                if self.multi_policy:
                    self.state_dim = state_shape.shape[0] 
                else:
                    extra_state_dim = 2 # 1: system indicator | 1: elap_time
                    self.state_dim = (state_shape.shape[0] * num_plant) + (extra_state_dim * num_plant)
            self.cont_action_dim = self.action_dim
            self.disc_action_dim = 1 # sf_len
        self.hybrid_action_dim = self.cont_action_dim + self.disc_action_dim

        self.vae_trainer = VAEModule(
            self.state_dim, self.hybrid_action_dim, self.latent_dim,
            self.max_action, vae_lr=vae_lr, hidden_size=vae_hidden_size, device=device,
            policy_name=self.policy_name, dep_check=dep_check, dec_explo=self.dec_explo,
            loss_dyn=loss_dyn, loss_energy=loss_energy, energy_coeff=kwargs['energy_coeff'],
        )

        # === replay buffer ===
        self.replay_buffer = ReplayBuffer(self.state_dim, self.hybrid_action_dim, device=device)

        # === policy ===
        self.policy = None
        # === scheduler ===
        self.random_scheduler = scheduler(algo='sequential', schedule_len=num_plant+1, max_period=max_period-1)
        # === eps_greedy ===
        self.eps_greedy = epsilon_decay(
            eps_decay=self.eps_decay, # 1e-5
            eps_final=0.2
        )
        # === other parameters ===
        self.learning_steps_vae = 0
        self.learning_steps_policy = 0
        self.step_cunt = 0
        self.all_state = None

        self.prev_state = None
        self.all_done = None
    

    def set_policy(self):
        if self.algo_name == 'plas_latent':
            if self.multi_policy:
                self.policy_list = [Latent(self.vae_trainer.vae, self.state_dim, self.hybrid_action_dim, self.latent_dim,
                self.max_action, device=self.device, policy_name=self.policy_name,
                dec_explo=self.dec_explo, nstep_return=self.nstep_return, 
                min_period=self.min_period, max_period=self.max_period,
                dtdiscount=self.dtdiscount, novae=self.novae,
                **self.kwargs
                ) for _ in range(self.num_plant)]
            else:
                self.policy = Latent(self.vae_trainer.vae, self.state_dim, self.hybrid_action_dim, self.latent_dim,
                    self.max_action, device=self.device, policy_name=self.policy_name,
                    dec_explo=self.dec_explo, nstep_return=self.nstep_return, 
                    min_period=self.min_period, max_period=self.max_period,
                    dtdiscount=self.dtdiscount, novae=self.novae,
                    **self.kwargs
                )
        else:
            raise NotImplementedError

    def init_state(self, env_list, ob):
        state = np.array([env.reset() for env in env_list], dtype=np.float32).reshape(self.num_plant, -1)
        self.prev_state = state
        ob.reset_episode(state)

        # === reference timer ===
        self.timer = {'global': 0}
        for idx in range(self.num_plant):
            self.timer[idx] = 0
        
        # === all system done ===
        self.all_done = np.zeros((self.num_plant,), dtype=np.float32)

    def random_slot_act(self, timer, syst_id):
        z = torch.zeros(self.latent_dim, dtype=torch.float32)
        caction = np.random.uniform(-1., 1., 1).astype(np.float32)

        
        if self.action_relabel:
            dt = np.random.randint(
                self.min_period, self.max_period+1, 
                size=(1,)
            )
            k = dts_to_k(dt, min_=self.min_period, max_=self.max_period)
        else:
            k = np.random.uniform(-1., 1., 1).astype(np.float32)
            dt = k_to_dts(k, min_=self.min_period, max_=self.max_period)

        adj_dt = schedulable_check(timer, syst_id, dt, self.num_plant, self.min_period, self.max_period)
        adj_k = dts_to_k(adj_dt, min_=self.min_period, max_=self.max_period)
        
        

        action = np.concatenate([caction, k], axis=-1)
        if dt != adj_dt:
            action = np.concatenate([caction, adj_k], axis=-1)
        return caction, adj_dt[0], action, z, action

    def slot_act(self, obs, timer, syst_id, eps=None, latent_explor=False):
        '''
        latent_explor=True : z를 랜덤하게 선택함
        '''
        if eps is not None:
            eps_ = self.eps_greedy.decay_eps(syst_id)

            if np.random.rand() < eps_:
                caction, dt, action, z, dummy_action = self.random_slot_act(timer, syst_id)
                return caction, dt, action, z, dummy_action

        obs = obs.reshape(-1)
        if self.multi_policy:
            action, z = self.policy_list[syst_id].select_action(obs, latent_explor=latent_explor)
        else:
            action, z = self.policy.select_action(obs, latent_explor=latent_explor)
        decode_action = deepcopy(action)
        
        caction, adj_dt = self.make_cont_disc_actions(action, timer, syst_id)
        
        
        return caction, adj_dt[0], action, z, decode_action


    def step(self, env_list, ob, t, is_random, delay_step_=False, loss_prob=0.):
        # print(f"===========timer:{self.timer}================")
        sf_reward = 0.
        terminal = False

        syst_id = select_syst(self.timer)
        curr_time = get_curr_time(self.timer)
        prev_state, prev_action, state, _reward, _done = ob.get_curr_transition(syst_id, curr_time)

        if self.multi_policy:
            prev_state = prev_state[syst_id]
            state = state[syst_id]
        
        # === memorize transition ===
        self.replay_buffer.add(prev_state.reshape(-1), prev_action, state.reshape(-1), _reward, _done)
                
        # ob.save_prev_state(syst_id) # for next transition save: 'state' to 'prev_state'
        # === select action ===
        if is_random:
            caction, dt, action, _, _ = self.random_slot_act(self.timer, syst_id)
        else:
            caction, dt, action, _, _ = self.slot_act(state, self.timer, syst_id, eps=self.eps_decay)
        # print(f"action_seq:{action_seq} | dts:{dts}")
        # print(f"action_seq[{syst_id}, {idx}]:{action_seq[syst_id, idx:idx+1]} | dt:{dt}")
        
        # === actuation packet loss ===
        prob_act = random.uniform(0., 1.)
        if prob_act < loss_prob:    # packet loss
            action = prev_action
            caction, adj_dt = self.make_cont_disc_actions(action, self.timer, syst_id)
            dt = adj_dt[0]
        
        # == step env ==
        next_state, reward_, done, _ = dt_step(
            env_list[syst_id],\
            caction, \
            dt=dt
        )
        # energy_reward = 1. * self.energy_coeff * np.linalg.norm((action[-1:]+1)/2, ord=2, axis=0, keepdims=True)   # energy save
        reward = reward_

        # === processing for next step ===
        sf_reward += reward
        self.timer[syst_id] += dt
        conflict = timer_check(self.timer, self.num_plant, max_step=1000)
        if conflict:
            print(f"conflict timer:{self.timer}")
        
        # === sensing packet loss ===
        prob_sens = random.uniform(0., 1.)
        if prob_sens >= loss_prob:    # packet success
            ob.append(syst_id, action, next_state, reward, done)
        
        if done:
            self.all_done[syst_id] = done

        # === memorize transition ===
        next_syst_id = select_syst(self.timer)
        save_next_state = deepcopy(ob.get_state(syst_id).reshape(-1)) if not self.multi_policy else deepcopy(ob.get_state(syst_id)[syst_id].reshape(-1))
        save_reward = sf_reward

        # === all system done ===
        if np.sum(self.all_done) >= self.num_plant:
            self.init_state(env_list, ob)
            terminal = True


        return t, reward, terminal, syst_id


    def evaluate_steps(self, env_list, ob, loss_prob=0.):
        self.init_state(env_list, ob)
        timer = {'global': 0}
        for idx in range(self.num_plant):
            timer[idx] = 0

        episode_return = 0.0
        syst_return = [0. for _ in range(self.num_plant)]
        all_done = np.zeros((self.num_plant,), dtype=np.float32)
        terminal = False
        syst_id = 0
        cont_cunt = 1
        syst_dt_ratio = {}
        trajectory = {
            'state':[],
            'action':[],
            'dt':[],
            'latent_action':[],
            'hybrid_action':[],
        }
        curr_time_list = []
        for i in range(self.num_plant):
            syst_dt_ratio[i] = [0 for _ in range(self.sf_len)]

        while not terminal:
            # print(f"timer:{timer}")
            # === select a system to schedule ===
            syst_id = select_syst(timer)
            curr_time = get_curr_time(timer)
            if curr_time > 0:
                cont_cunt += 1
            # state = deepcopy(ob.get_state(syst_id)) if not self.multi_policy else deepcopy(ob.get_state(syst_id)[syst_id])
            _, prev_action, state_, _, _ = ob.get_curr_transition(syst_id, curr_time)
            state = deepcopy(state_)
            if self.multi_policy:
                state = state[syst_id]
            # === select action ===
            if self.eval_random_act == 'origin':
                caction, dt, action, z, decode_action = self.random_slot_act(timer, syst_id)
            elif self.eval_random_act == 'latent':
                caction, dt, action, z, decode_action = self.slot_act(state, timer, syst_id, latent_explor=True)
            else:
                caction, dt, action, z, decode_action = self.slot_act(state, timer, syst_id)
            
            # === actuating packet loss ===
            prob = random.uniform(0., 1.)
            if prob < loss_prob:    # packet loss
                action = prev_action
                caction, adj_dt = self.make_cont_disc_actions(action, self.timer, syst_id)
                dt = adj_dt[0]
            
            # === env step ===
            next_state, reward, done, _ = dt_step(env_list[syst_id], caction, dt=dt)

            # === processing for next step ===
            timer[syst_id] += dt
            conflict = timer_check(timer, self.num_plant, max_step=1000)
            if conflict:
                print(f"conflict timer:{timer}")
            syst_return[syst_id] += reward
            
            # === sensing packet loss ===
            prob_sens = random.uniform(0., 1.)
            if prob_sens >= loss_prob:    # packet success
                ob.append(syst_id, action, next_state, reward, done)
            if done:
                all_done[syst_id] = done

            if np.sum(all_done) >= self.num_plant:
                terminal = True
            # === count dt_ratio ===
            if dt <= self.max_period:
                syst_dt_ratio[syst_id][dt-1] += 1
            # === log trajectory ===
            if syst_id == 0:
                if self.multi_policy:
                    trajectory['state'].append(state)
                else:
                    trajectory['state'].append(state[0])
                trajectory['action'].append(caction)
                trajectory['dt'].append(dt)
                trajectory['latent_action'].append(z)
                # trajectory['hybrid_action'].append(action)
                trajectory['hybrid_action'].append(decode_action)
        episode_return = sum(syst_return)
        trajectory['state'] = np.stack(trajectory['state'], axis=0)
        trajectory['action'] = np.stack(trajectory['action']).squeeze()
        trajectory['dt'] = np.stack(trajectory['dt'])
        trajectory['latent_action'] = np.stack(trajectory['latent_action']).squeeze()
        trajectory['hybrid_action'] = np.stack(trajectory['hybrid_action']).squeeze()
        return episode_return, syst_dt_ratio, syst_return, trajectory, cont_cunt

    def make_cont_disc_actions(self, action, timer, syst_id):
        '''
        :param action: size [2,]: normalized [control command, transmission period]
        :param timer: simulator timer
        :param syst_id: one of multiple physical system ids
        
        :return 
        '''
        # === split command and period ===
        caction = action[:self.cont_action_dim]
        k = action[self.cont_action_dim:]
        dt = k_to_dts(k, min_=self.min_period, max_=self.max_period)   # [1]
        # === schedulable check ===
        adj_dt = schedulable_check(timer, syst_id, dt, self.num_plant, self.min_period, self.max_period)
        adj_k = dts_to_k(adj_dt, min_=self.min_period, max_=self.max_period)
        if self.action_relabel:
            action[self.cont_action_dim:] = adj_k
        else:
            if dt != adj_dt:
                action[self.cont_action_dim:] = adj_k
    
        return caction, adj_dt
    
    def vae_train(self, model_dir, itr, writer):
        self.learning_steps_vae += 1

        vae_loss, recon_loss, kl_loss, dyn_loss = self.vae_trainer.train(self.replay_buffer, model_dir, iterations=itr)
        

    def policy_train(self, itr, writer, syst_id=None):
        if self.eval_random_act is not None:
            return
        # elif self.exploration_check == True:
        #     return 
        self.learning_steps_policy += 1

        if self.multi_policy and syst_id is not None:
            # for syst_id in range(self.num_plant):
            critic_loss, actor_loss = self.policy_list[syst_id].train(self.replay_buffer, iterations=itr)
        else:
            critic_loss, actor_loss = self.policy.train(self.replay_buffer, iterations=itr)

        if self.learning_steps_policy % 100 == 0:
            writer.add_scalar("loss/critic", critic_loss.item(), self.learning_steps_policy)
            writer.add_scalar("loss/actor", actor_loss.item(), self.learning_steps_policy)