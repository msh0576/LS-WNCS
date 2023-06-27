import os
from collections import deque
from datetime import timedelta
from re import I
from time import sleep, time

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
from copy import deepcopy

from Models.util_model import dts_to_k

class MultiPlasObservation:
    """
    Observation for SLAC.
    """

    def __init__(self, state_size, action_size, num_plant, extra_state=0, multipolicy=False):
        self.state_size = state_size
        self.action_size = action_size
        self.num_plant = num_plant
        self.extra_state = extra_state
        self.multipolicy = multipolicy
        self.prev_time = 0
        self.min_elap_time = 0
        self.max_elap_time = 30


    def reset_episode(self, state=None):
        self._prev_state = np.zeros((self.num_plant, self.num_plant, self.state_size+self.extra_state), dtype=np.float32)
        self._state = np.zeros((self.num_plant, self.state_size), dtype=np.float32)
        self._tmp_state = np.zeros((self.num_plant, self.state_size), dtype=np.float32)
        self._action = np.zeros((self.num_plant, self.action_size), dtype=np.float32)
        self._reward = np.zeros((self.num_plant,), dtype=np.float32)
        # self._reward = [deque(maxlen=1) for _ in range(self.num_plant)]
        self._done = np.zeros((self.num_plant,), dtype=np.float32)
        self._elap_time = np.zeros((self.num_plant, 1), dtype=np.float32)
        self._elap_time_unnorm = np.zeros((self.num_plant, 1), dtype=np.float32)

        # for reward in self._reward:
        #     reward.append(np.zeros(1, dtype=np.float32))

        if state is not None:
            assert self._state.shape == state.shape
            self._state = state

    def append(self, syst_id, action, state, reward, done):
        self._action[syst_id] = action
        # self._state[syst_id] = state
        self._tmp_state[syst_id] = state
        # self._reward[syst_id] += reward
        self._reward[syst_id] = reward
        self._done[syst_id] = done

    def get_curr_transition(self, syst_id, curr_time):
        '''
        Output:
            state: [D_plant, D_state]
            action: [D_action,]
            reward: [1, ]
        '''
        self.step_elap_time(syst_id, curr_time) # count state elapsed time after last receiving
        self.set_curr_state(syst_id)    # Bring the already stored next_state to the current state
        prev_state = deepcopy(self._prev_state[syst_id])
        action = self._action[syst_id]
        reward = self._reward[syst_id]
        done = self._done[syst_id]
        state = self.get_state(syst_id)
        # for next transition save: 'state' to 'prev_state'
        self.save_prev_state(syst_id)
        return prev_state, action, state, reward, done

    def save_prev_state(self, syst_id):
        self._prev_state[syst_id] = deepcopy(self.get_state(syst_id))    # full-state

    def set_curr_state(self, syst_id):
        '''
        Bring the already stored next_state
        '''
        self._state[syst_id] = deepcopy(self._tmp_state[syst_id])


    def set_action(self, syst_id, action):
        self._action[syst_id] = action

    def get_state(self, syst_id):
        if self.multipolicy:
            state = self._state
        else:
            syst_flag = np.zeros((self.num_plant,1), dtype=np.float32)
            syst_flag[syst_id] = 1
            state = np.concatenate([self._state, syst_flag, self._elap_time], axis=-1)
            # self._elap_time[syst_id] = 0

        return state
    
    def get_action(self, syst_id):
        return self._action[syst_id]


    def get_elap_time(self, syst_id):
        return self._elap_time[syst_id]
    
    def step_elap_time(self, syst_id, time):
        dt = time - self.prev_time
        self._elap_time_unnorm += int(dt)
        self._elap_time_unnorm[syst_id] = 0
        # === normalize ===
        assert self._elap_time_unnorm.all() <= self.max_elap_time, 'should extend the max_elap_time'
        self._elap_time = dts_to_k(self._elap_time_unnorm, min_=self.min_elap_time, max_=self.max_elap_time)
        
        self.prev_time = time
        # print(f"self._elap_time_unnorm:{self._elap_time_unnorm}")
        # print(f"self._elap_time:{self._elap_time}")



    def get_done(self, syst_id):
        if self._done.sum() > 1.0:
            result = True
        else:
            result = False
        return result
    
    @property
    def state(self):
        return self._state
    
    @property
    def action(self):
        return self._action
    
    @property
    def reward(self):
        return self._reward
    
    @property
    def done(self):
        return self._done


class TrainerLSWNCS():
    def __init__(
        self,
        env,
        env_test,
        algo,
        log_dir,
        device,
        vae_itr,
        num_plant=7,
        seed=0,
        delay_env=False,
        num_steps=3 * 10 ** 6,
        initial_collection_steps=10 ** 4,
        initial_learning_steps=10 ** 5,
        initial_episodes=100,
        num_episodes=250,
        num_sequences=8,
        eval_interval=10 ** 4,
        eval_interval_epi = 10,
        num_eval_episodes=10,
        render=False,
        algo_name='slac',
        **kwargs,
    ):
        self.env_list = env
        self.env_test_list = env_test
        self.state_size = self.env_list[0].observation_space.shape[0]
        self.caction_size = self.env_list[0].action_space.shape[0]
        self.vae_itr = vae_itr
        self.num_plant = num_plant
        self.delay_env = delay_env
        self.device = device
        self.num_steps = num_steps * num_plant
        self.initial_collection_steps = initial_collection_steps * num_plant
        self.initial_learning_steps = initial_learning_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.render = render
        self.initial_episodes = initial_episodes
        self.num_episodes = num_episodes
        self.eval_interval_epi = eval_interval_epi
        self.loss_prob = 0.001 # 0.1% packet loss

        [env_.seed(seed) for env_ in self.env_list]
        [env_test_.seed(2 ** 31 - seed) for env_test_ in self.env_test_list]

        hybrid_action_dim = self.caction_size + 1
        if algo.multi_policy:
            extra_state_dim = 0 
        else:
            extra_state_dim = 2 # 1: system indicator | 1: elap_time
        self.ob = MultiPlasObservation(self.state_size, hybrid_action_dim, self.num_plant, extra_state_dim, algo.multi_policy)
        self.ob_test = MultiPlasObservation(self.state_size, hybrid_action_dim, self.num_plant, extra_state_dim, algo.multi_policy)

        # Algorithm to learn.
        self.algo = algo
        self.algo_name = algo_name
        
        

        # Log setting.
        self.fieldnames = ['step', 'return', 'dt_ratio', 'syst_return', 'trajectory', 'std_return', 'cont_cunt']
        self.log = {"step": [], "return": [], "dt_ratio":[], "syst_return":[], "trajectory":[], "std_return":[], "cont_cunt":[]}
        self.csv_path = os.path.join(log_dir, "log.csv")
        self.log_dir = log_dir
        self.summary_dir = os.path.join(log_dir, "summary")
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, "model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def episode_train(self):    # 한 episode 단위로 평가, 실행
        self.start_time = time()
        t = 0
        self.algo.init_state(self.env_list, self.ob)

        # === Collect trajectories using random policy. ===
        print(f"[Initial collection steps]")
        for epi in tqdm(range(1, self.initial_episodes + 1)):
            done = False
            while not done:
                t, _ , done, _ = \
                    self.algo.step(
                        self.env_list, self.ob, t, True,
                        delay_step_=self.delay_env,
                        loss_prob=self.loss_prob
                    )

        # === set policy ===
        self.algo.set_policy()
        # === train VAE ===
        print("Training VAE...")
        self.algo.vae_train(self.model_dir, self.vae_itr, writer=self.writer)

        # === Iterate collection, update and evaluation. ===
        episode_iter = 0
        num_policy_train = 0

        for epi in tqdm(range(self.initial_episodes + 1, self.num_episodes)):
            done = False
            episode_reward = 0

            while not done:
                t, reward , done, syst_id = self.algo.step(
                    self.env_list, self.ob, t, False,
                    delay_step_=self.delay_env,
                    loss_prob=self.loss_prob
                    
                )

                episode_reward += reward
                
                
                # === state and action distribution (exploration) 결과 확인 용
                if self.algo.exploration_check and (num_policy_train == 0 or num_policy_train == 100000):
                    self.evaluate(num_policy_train, self.num_eval_episodes*10, exploration_check=self.algo.exploration_check)
                
                # === update policy and VAE ===
                self.algo.policy_train(itr=5, writer=self.writer, syst_id=syst_id)
                
                num_policy_train += 1
                


            # === evaluation ===
            if epi % self.eval_interval_epi == 0:
                self.evaluate(epi, self.num_eval_episodes)
            


    


    def evaluate(self, step_env, num_eval_episodes, exploration_check=False):
        '''
        exploration_check = True : at a specific 'step_env', logs 'hybrid_action' trajectory
        '''
        mean_return = 0.0
        mean_cont_cunt = 0
        mean_dt_ratio = {}
        all_syst_dt_ratio = [0 for _ in range(self.algo.sf_len)]
        mean_syst_return = [0. for _ in range(self.algo.num_plant)]
        episode_return_list = []
        episode_state_traj = []
        episode_hybrid_action_traj = []
        episode_latent_action_traj = []
        episode_dt_traj = []
        state_traj_min_len = 1000000

        for i in range(num_eval_episodes):
            episode_return, syst_dt_ratio, syst_return, trajectory, cont_cunt = \
                self.algo.evaluate_steps(
                    self.env_test_list, self.ob_test,
                    loss_prob=self.loss_prob
                )

            # === preprocess for logging ===
            episode_return_list.append(episode_return)
            mean_return += episode_return / num_eval_episodes
            mean_cont_cunt += cont_cunt / num_eval_episodes
            # for state and action exploration check
            episode_state_traj.append(trajectory['state'])
            episode_hybrid_action_traj.append(trajectory['hybrid_action'])
            episode_latent_action_traj.append(trajectory['latent_action'])
            episode_dt_traj.append(trajectory['dt'])
            
            if trajectory['state'].shape[0] < state_traj_min_len:
                state_traj_min_len = trajectory['state'].shape[0]
            
            for idx in range(self.num_plant):
                mean_syst_return[idx] += syst_return[idx] / num_eval_episodes

            for syst_id, dt_ratio in syst_dt_ratio.items():
                if i == 0:
                    mean_dt_ratio[syst_id] = [0 for _ in range(len(syst_dt_ratio[syst_id]))]
                for j in range(len(dt_ratio)):
                    mean_dt_ratio[syst_id][j] += dt_ratio[j]
                    all_syst_dt_ratio[j] += dt_ratio[j]
        mean_return = mean_return / self.num_plant
        std_return = np.std(np.array(episode_return_list)) / self.num_plant
        print(f'eps:{self.algo.eps_greedy.eps}')
        print(f"mean_cont_cunt:{mean_cont_cunt}")
        # print(f"std_return{std_return}")
        
        # === state exploration preprocessing ===
        # === print discrete action ratio ===
        print("<<dt ratio>>")
        total_action = sum(all_syst_dt_ratio)
        for i in range(self.algo.sf_len):
            print("[{}] {}% | ".format(i, round((all_syst_dt_ratio[i]/total_action)*100, 1)), end='')
        print("")
        


        # Log to CSV.
        if not exploration_check:
            self.log["step"].append(step_env)
            self.log["return"].append(mean_return)
            self.log["dt_ratio"].append(mean_dt_ratio)
            self.log["syst_return"].append(mean_syst_return)
            self.log["trajectory"].append(trajectory)
            self.log["std_return"].append(std_return)
            self.log["cont_cunt"].append(mean_cont_cunt)
            pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

            # ===== Log to CSV with dictionary =====
            np.savetxt(os.path.join(self.log_dir, 'state_log.csv'), trajectory['state'], delimiter=',')
            np.savetxt(os.path.join(self.log_dir, 'dt_log.csv'), trajectory['dt'], delimiter=',')
            np.savetxt(os.path.join(self.log_dir, 'action_log.csv'), trajectory['action'], delimiter=',')
            np.savetxt(os.path.join(self.log_dir, 'latent_action_log.csv'), trajectory['latent_action'], delimiter=',')
        
        for idx, state_traj in enumerate(episode_state_traj):
            if exploration_check:
                np.savetxt(os.path.join(self.log_dir, f'envstep-{step_env}_episode-{idx}_state_log.csv'), state_traj, delimiter=',')
                np.savetxt(os.path.join(self.log_dir, f'envstep-{step_env}_episode-{idx}_hybrid_action_log.csv'), episode_hybrid_action_traj[idx], delimiter=',')
                np.savetxt(os.path.join(self.log_dir, f'envstep-{step_env}_episode-{idx}_latent_action_log.csv'), episode_latent_action_traj[idx], delimiter=',')
                np.savetxt(os.path.join(self.log_dir, f'envstep-{step_env}_episode-{idx}_dt_log.csv'), episode_dt_traj[idx], delimiter=',')
            else:
                np.savetxt(os.path.join(self.log_dir, f'episode-{idx}_state_log.csv'), state_traj, delimiter=',')
                np.savetxt(os.path.join(self.log_dir, f'episode-{idx}_hybrid_action_log.csv'), episode_hybrid_action_traj[idx], delimiter=',')
                np.savetxt(os.path.join(self.log_dir, f'episode-{idx}_latent_action_log.csv'), episode_latent_action_traj[idx], delimiter=',')
                np.savetxt(os.path.join(self.log_dir, f'episode-{idx}_dt_log.csv'), episode_dt_traj[idx], delimiter=',')
                

        # Log to TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step_env)
        print(f"Steps: {step_env:<6}   | " f"Return: {mean_return:<5.1f}  | " f"Time: {self.time}  | ", end='')
        for idx in range(self.num_plant):
            print(f"syst_{idx} return: {mean_syst_return[idx]:<5.1f}  |  ", end='')
        print("")



    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))