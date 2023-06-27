import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from Models.sac.network import Policy_SAC, SoftQNetwork, layer_init, soft_update
from Models.sac.buffer import ReplayBuffer
from Models.delay_sac.env import delay_step, get_delay, dt_step
from Models.functions import scheduler
from Models.util_model import dts_to_k, k_to_dts

class SacAlgorithm:
    def __init__(self, 
        buffer_size, 
        batch_size, 
        input_shape, 
        out_c, 
        device, 
        max_period,
        sf_len,
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
        pkt_loss=0.7,
        num_plant=7,
    ):
        self.batch_size = batch_size
        self.device = device
        self.autotune = autotune
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau
        self.gamma = gamma
        self.min_period = 1
        self.max_period = max_period
        self.sf_len = sf_len
        self.use_hybrid_action = True
        self.exploration_check = True # False: default | True: 초기 샘플 수집 후 policy training 없이 state and action exploration (distribution)이 어떻게 나오는지 확인하기 위해


        print(f"use_hybrid_action:{self.use_hybrid_action} | exploration_check:{self.exploration_check}")
        # self.hybrid_action_dim = out_c*(sf_len//self.min_period)+1
        if self.use_hybrid_action:
            self.hybrid_action_dim = out_c + 1
        else:
            self.hybrid_action_dim = out_c


        # print("input_shape:{} | out_c:{}".format(input_shape, out_c))

        self.buffer = ReplayBuffer(buffer_size)
        self.pg = Policy_SAC(input_shape, self.hybrid_action_dim, device).to(device)
        self.qf1 = SoftQNetwork(input_shape, self.hybrid_action_dim, layer_init, device).to(device)
        self.qf2 = SoftQNetwork(input_shape, self.hybrid_action_dim, layer_init, device).to(device)
        self.qf1_target = SoftQNetwork(input_shape, self.hybrid_action_dim, layer_init, device).to(device)
        self.qf2_target = SoftQNetwork(input_shape, self.hybrid_action_dim, layer_init, device).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.values_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr)
        self.policy_optimizer = optim.Adam(list(self.pg.parameters()), lr=policy_lr)
        self.loss_fn = nn.MSELoss()

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
        
        # network scheduler
        self.random_scheduler = scheduler(algo='sequential', schedule_len=num_plant)

        # Other parameters
        self.prev_state = None  
        self.learning_steps_sac = 0
        self.prev_action = None
        self.pkt_loss = pkt_loss
        self.num_plant = num_plant
        self.dtdiscount = 0.99

    
    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        # torch.save({
        #     'actor_model': self.pg.state_dict(),
        #     'critic_model1': self.qf1.state_dict(),
        #     'critic_model2': self.qf2.state_dict()
        # }, path)
        torch.save(self.pg.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.qf1.state_dict(), os.path.join(path, "critic1.pth"))
        torch.save(self.qf2.state_dict(), os.path.join(path, "critic2.pth"))
    
    def load_model(self, path):
        learned_model = torch.load(path)
        # self.pg.load_state_dict(learned_model['actor_model'])
        # self.qf1.load_state_dict(learned_model['critic_model1'])
        # self.qf2.load_state_dict(learned_model['critic_model2'])
        self.pg.load_state_dict(os.path.join(path, "actor.pth"))
        self.qf1.load_state_dict(os.path.join(path, "critic1.pth"))
        self.qf2.load_state_dict(os.path.join(path, "critic2.pth"))

    def preprocess(self, ob):
        state = torch.tensor(ob.state, dtype=torch.float, device=self.device).view(-1)    #[D_state]
        return state

    def explore(self, obs):
        state = self.preprocess(obs)
        action, _ = self.pg.get_action(state)
        return action.detach().cpu().numpy()
    
    def exploit(self, obs):
        state = self.preprocess(obs)
        return self.pg.exploit(state)[0].detach().cpu().numpy()
    
    def step(self, env, ob, t, is_random, delay_step_=False):

        syst_idx = 0
        dt = int(self.random_scheduler.get_period(syst_idx))

        t += 1
        if is_random:
            action = env.action_space.sample()
        else:
            action = self.explore(ob)
        # print(f"dt:{dt} | action:{action}")
        if not delay_step_:
            # state, reward, done, _ = env.step(action)
            state, reward, done, _ = dt_step(env, action, dt)
        else:
            # == action and observation delay
            act_delay =  min(get_delay(self.pkt_loss), dt-1)
            obs_delay =  min(get_delay(self.pkt_loss), dt-1)
            # == env step ==
            state, reward, done, _  = delay_step(env, action, self.prev_action, dt, act_delay, obs_delay)
        mask = False if t == env.spec.max_episode_steps else done
        ob.append(state, action)
        self.buffer.put((self.prev_state, action, reward, state, done))
        self.prev_state = state
        self.prev_action = action

        if done:
            t = 0
            state = env.reset()
            ob.reset_episode(state)
            self.prev_state = state
            self.prev_action = None
            # self.buffer.reset_episode(state)

        return t, reward, done
    
    
    
    
    def evaluate_steps(self, env, ob, delay_step_=False, dt_=None):
        state = env.reset()
        ob.reset_episode(state)
        episode_return = 0.0
        done = False
        syst_idx = 0
        prev_action = env.action_space.sample()
        dt_ratio = [0 for _ in range(self.sf_len+1)]
        # dt_ratio = [0 for _ in range(self.num_plant+1)]
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
            if dt_ is not None:
                dt = dt_
            else:
                dt = int(self.random_scheduler.get_period(syst_idx))
            action = self.exploit(ob)
            # print(f"dt:{dt} | action:{action}")
            if not delay_step_:
                # state, reward, done, _ = env.step(action)
                state, reward, done, _ = dt_step(env, action, dt)
            else:
                # == action and observation delay
                act_delay =  min(get_delay(self.pkt_loss), dt-1)
                obs_delay =  min(get_delay(self.pkt_loss), dt-1)
                # == env step ==
                state, reward, done, _  = delay_step(env, action, prev_action, dt, act_delay, obs_delay)
            ob.append(state, action)
            episode_return += reward
            prev_action = action
            dt_ratio[dt-1] += 1

        return episode_return, dt_ratio, trajectory, cont_cunt

    
    def random_action(self):
        caction = np.random.uniform(-1., 1., 1).astype(np.float32)
        dt = np.random.randint(
            self.min_period, self.max_period+1, 
            size=(1,)
        )
        k = dts_to_k(dt, min_=self.min_period, max_=self.max_period)

        action = np.concatenate([caction, k], axis=-1)

        return caction, dt[0], action
    
    def slot_act(self, obs):
        action = self.explore(obs)
        caction = action[:1]
        k = action[1:]
        dt = k_to_dts(k, min_=self.min_period, max_=self.max_period)   # [1]
        return caction, dt[0], action
                
    
    def hybrid_step(self, env, ob, t, is_random, delay_step_=False):
        syst_idx = 0

        t += 1
        if is_random:
            caction, dt, action = self.random_action()
        else:
            caction, dt, action = self.slot_act(ob)
        
        state, reward, done, _ = dt_step(env, caction, dt)
        
        mask = False if t == env.spec.max_episode_steps else done
        ob.append(state, action)
        self.buffer.put((self.prev_state, action, reward, state, done))
        self.prev_state = state
        self.prev_action = action

        if done:
            t = 0
            state = env.reset()
            ob.reset_episode(state)
            self.prev_state = state
            self.prev_action = None
            # self.buffer.reset_episode(state)

        return t, reward, done

    def evaluate_hybrid_steps(self, env, ob, delay_step_=False, dt_=None):
        state = env.reset()
        ob.reset_episode(state)
        episode_return = 0.0
        done = False
        syst_idx = 0
        prev_action = env.action_space.sample()
        dt_ratio = [0 for _ in range(self.sf_len+1)]
        # dt_ratio = [0 for _ in range(self.num_plant+1)]
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
            caction, dt, action = self.slot_act(ob)

            state, reward, done, _ = dt_step(env, caction, dt)
            ob.append(state, action)
            episode_return += reward
            prev_action = action
            dt_ratio[dt-1] += 1
            
            # === log trajectory ===
            trajectory['state'].append(state)
            trajectory['action'].append(caction)
            trajectory['dt'].append(dt)
            trajectory['hybrid_action'].append(action)

        return episode_return, dt_ratio, trajectory, cont_cunt



    def superframe(self):
        dt = np.random.randint(self.min_period, self.max_period+1)

        dts = [dt for _ in range(self.sf_len//dt)]
        if self.sf_len - sum(dts) > 0:
            dts.append(self.sf_len - sum(dts))
        # print(f"dt:{dt} | dts:{dts}")
        return dts

    def sf_act(self, obs):
        if type(obs) != torch.tensor:
            obs = torch.tensor(obs, dtype=torch.float, device=self.device).view(-1)    #[D_state]
        obs = obs.unsqueeze(0)
        action, _ = self.pg.get_action(obs)
        action = action.cpu().data.numpy().flatten()
        # print(f"action:{action}")

        # === static period during a superframe ===
        k_norm = action[-1:]
        k = self.k_unnormal(k_norm[0])
        # === periods during a SF ===
        # if k == 0:
        #     print(f"k:{k} | k_norm:{k_norm}")
        dts = [k] * (self.sf_len // k )    
        dt_resid = self.sf_len - sum(dts)
        if dt_resid > 0:
            dts.append(dt_resid)
        return action, dts



    def sf_step(self, env, ob, t, is_random, delay_step_=False):
        sf_reward = 0.
        action_vec_list = []

        # == superframe period ==
        dts = self.superframe()

        if not is_random:
            action_vec, dts = self.sf_act(self.prev_state)
            # print(f"dts:{dts}")
        for idx, dt in enumerate(dts):
            # == select continuous action ==
            if is_random:
                caction = env.action_space.sample()
                if len(action_vec_list) < self.hybrid_action_dim - 1:
                    action_vec_list.append(caction)
            else:
                if idx >= len(action_vec):
                    caction = action_vec[-1:]
                else:
                    caction = action_vec[idx:idx+1]
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
            action_vec = np.concatenate([self.pad_action(action_vec_numpy), k])
        else:
            # if generated action_vec elements are not used, it is replaced to 0
            if idx < len(action_vec) - 2:
                action_vec[idx+1:-1] = 0.
            # print(f"dts:{dts} | action_vec;{action_vec}")
        
        # == memorize transition ==
        self.memory_push(self.prev_state, action_vec, sf_reward, state, done)
        # == env postprocess ==
        self.prev_state = state
        

        return t, reward, done


    def evaluate_sf_steps(self, env, ob, delay_step_=False, dt_=None):
        state = env.reset()
        prev_state = state
        episode_return = 0.0
        done = False
        dt_ratio = [0 for _ in range(self.sf_len+1)]

        while not done:
            
            action_vec, dts = self.sf_act(prev_state)
            # print(f"dts:{dts} | action_vec:{action_vec}")
            for idx, dt in enumerate(dts):
                # == select continuous action ==
                if idx >= len(action_vec):
                    action = action_vec[-1:]
                else:
                    action = action_vec[idx:idx+1]

                state, reward, done, _ = dt_step(env, action, dt)
                episode_return += reward

                if done:
                    break
            dt_ratio[dts[0]-1] += 1
            prev_state = state

        return episode_return, dt_ratio


    def train(self, global_step, writer):
        # if self.exploration_check:  # state and action distribution 체크할 때는 학습 안한상태에서 확인
        #     return 
        
        if len(self.buffer.buffer) > self.batch_size:  # starts update as soon as there is enough data.
            self.learning_steps_sac += 1

            s_obs, s_actions, s_rewards, s_next_obses, s_dones = self.buffer.sample(self.batch_size)

            s_obs = torch.tensor(s_obs, dtype=torch.float32, device=self.device)    # [B, D_state]
            s_next_obses = torch.tensor(s_next_obses, dtype=torch.float32, device=self.device)
            s_actions = torch.tensor(s_actions, dtype=torch.float32, device=self.device)
            s_rewards = torch.tensor(s_rewards, dtype=torch.float32, device=self.device)               # [B, ]
            s_dones = torch.tensor(s_dones, dtype=torch.float32, device=self.device)     # [B, ]

            # === calculate dt ===
            k = s_actions[:, -1:].detach().cpu().numpy() # [B, 1]
            dt = torch.tensor(k_to_dts(k, self.min_period, self.max_period), dtype=torch.int32, device=self.device)
            
            # === critic udpate ===
            # print("obs:{} | action:{} | rewards:{} | dones:{}".format(s_obs.shape, s_actions.shape, s_rewards.shape, s_dones.shape))
            with torch.no_grad():
                next_state_actions_c, next_state_log_pi_c = self.pg.get_action(s_next_obses)
                qf1_next_target = self.qf1_target.forward(s_next_obses, next_state_actions_c)
                qf2_next_target = self.qf2_target.forward(s_next_obses, next_state_actions_c)

                # print(f"next_state_actions_c:{next_state_actions_c.shape}")
                # print("qf1_next_target:{} | next_state_log_pi_c:{} ".format(qf1_next_target.shape,  next_state_log_pi_c.shape))
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi_c  # [B, 1]
                next_q_value = (s_rewards + (1 - s_dones) * self.gamma * min_qf_next_target.view(-1)).unsqueeze(1) # [B, 1]
                # print(f"next_q_value:{next_q_value.shape}")
                
                # === discounted reward method ===
                if self.use_hybrid_action:
                    next_q_value = (self.dtdiscount ** dt * s_rewards.unsqueeze(1)) + (1 - s_dones).unsqueeze(1) * self.gamma * min_qf_next_target  # [B, 1]
                

            qf1_a_values = self.qf1.forward(s_obs, s_actions)
            qf2_a_values = self.qf2.forward(s_obs, s_actions)
            qf1_loss = self.loss_fn(qf1_a_values, next_q_value)
            qf2_loss = self.loss_fn(qf2_a_values, next_q_value)
            qf_loss = (qf1_loss + qf2_loss) / 2

            self.values_optimizer.zero_grad()
            qf_loss.backward()
            self.values_optimizer.step()

            # === actor update ===
            if global_step % self.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                        self.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    actions_c, log_pi_c = self.pg.get_action(s_obs)
                    qf1_pi = self.qf1.forward(s_obs, actions_c)
                    qf2_pi = self.qf2.forward(s_obs, actions_c)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)

                    policy_loss = (self.alpha * log_pi_c - min_qf_pi).mean()

                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()


            # update the target network
            if global_step % self.target_network_frequency == 0:
                for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # === log ===
            if self.learning_steps_sac % 1000 == 0:
                writer.add_scalar("loss/critic", qf_loss.item(), self.learning_steps_sac)
                writer.add_scalar("loss/critic", policy_loss.item(), self.learning_steps_sac)


    
    def memory_push(self, obs, action, reward, next_obs, done):
        self.buffer.put((obs, action, reward, next_obs, done))
    
    def pad_action(self, action_vec):
        if len(action_vec) == self.hybrid_action_dim-1:
            return action_vec
        pad = np.zeros(self.hybrid_action_dim-1 - len(action_vec), dtype=np.float)
        pad_action_vec = np.concatenate([action_vec, pad])
        return pad_action_vec
    
    def k_unnormal(self, k_norm):
        # return round(k_norm*(self.max_period - self.min_period) + self.min_period)
        return round( (k_norm+1) / 2 * (self.max_period - self.min_period) + self.min_period)
    
    def k_normal(self, k):
        # return np.array([(k - self.min_period) / (self.max_period - self.min_period)], dtype=np.float)
        return np.array([2 * (k - self.min_period) / (self.max_period - self.min_period) - 1], dtype=np.float)