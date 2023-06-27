import os
from collections import deque
from datetime import timedelta
from ossaudiodev import SNDCTL_DSP_RESET
from re import I
from time import sleep, time

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt


from Models.delay_sac.trainer import Trainer

class TrainerSac(Trainer):
    def __init__(
        self,
        vae_itr,
        initial_episodes=100,
        num_episodes=250,
        eval_interval_epi = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        algo_name = kwargs['algo_name']
        sf_sched = kwargs['sf_sched']
        self.log['std_return']  = []
        self.log["cont_cunt"] = []
        self.initial_episodes = initial_episodes
        self.num_episodes = num_episodes
        self.eval_interval_epi = eval_interval_epi
        
        if algo_name == 'hsac' and sf_sched:
            self.sched_step = self.algo.sf_step
            self.eval_sched_step = self.algo.evaluate_sf_steps
        else:
            self.sched_step = self.algo.step
            self.eval_sched_step = self.algo.evaluate_steps
            
            if self.algo.use_hybrid_action:
                self.sched_step = self.algo.hybrid_step
                self.eval_sched_step = self.algo.evaluate_hybrid_steps
            
        self.motivation = False
        print(f"motivation:{self.motivation}")
        if self.motivation:
            self.log['motivation_return'] = []

    def episode_train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()
        self.ob.reset_episode(state)
        self.algo.prev_state = state
        self.algo.prev_action = self.env.action_space.sample()
        
        # === Collect trajectories using random policy. ===
        print(f"[Initial collection steps]")
        for epi in tqdm(range(1, self.initial_episodes + 1)):
            done = False
            while not done:
                # t, _ , done, _ = \
                #     self.algo.step(
                #         self.env_list, self.ob, t, True,
                #         delay_step_=self.delay_env
                #     )
                t, _ , done = self.sched_step(self.env, self.ob, t, True, delay_step_=self.delay_env)
                

        # === Iterate collection, update and evaluation. ===
        episode_iter = 0
        num_policy_train = 0

        for epi in tqdm(range(self.initial_episodes + 1, self.num_episodes)):
            done = False
            episode_reward = 0

            while not done:
                # === environment step ===
                t, reward , done = self.sched_step(self.env, self.ob, t, False, delay_step_=self.delay_env)

                episode_reward += reward
                
                
                # === state and action distribution (exploration) 결과 확인 용
                if self.algo.exploration_check and (num_policy_train == 0 or num_policy_train == 100000):
                    self.evaluate(num_policy_train, self.num_eval_episodes*10, exploration_check=self.algo.exploration_check)
                
                # === update policy ===
                self.algo.train(0, self.writer)
                
                num_policy_train += 1
                

            # === evaluation ===
            if epi % self.eval_interval_epi == 0:
                if self.motivation:
                    self.motivation_evaluate(epi)
                else:
                    self.evaluate(epi, self.num_eval_episodes)
        
    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()
        self.ob.reset_episode(state)
        self.algo.prev_state = state
        self.algo.prev_action = self.env.action_space.sample()

        # === Collect trajectories using random policy. ===
        print(f"[Initial collection steps]")
        for step in tqdm(range(1, self.initial_collection_steps + 1)):
            t, _ , _ = self.sched_step(self.env, self.ob, t, step <= self.initial_collection_steps, delay_step_=self.delay_env)
        

        # === Iterate collection, update and evaluation. ===
        episode_reward = 0
        episode_iter = 0

        # for step in tqdm(range(self.initial_collection_steps + 1, self.num_steps // (self.num_plant+1) + 1)):
        for step in tqdm(range(self.initial_collection_steps + 1, self.num_steps)):
            # === environment step ===
            t, reward , done = self.sched_step(self.env, self.ob, t, False, delay_step_=self.delay_env)

            episode_reward += reward
            if done:
                episode_reward = 0
                episode_iter += 1
            # === update policy and VAE ===
            self.algo.train(step, self.writer)

            # === evaluation ===
            step_env = step * (self.num_plant+1)
            if step % self.eval_interval == 0:
                if self.motivation:
                    self.motivation_evaluate(step)
                else:
                    self.evaluate(step)
                self.algo.save_model(self.model_dir)

    def evaluate(self, step_env, num_eval_episodes, exploration_check=False):
        mean_return = 0.0
        mean_dt_ratio = [0 for _ in range(self.algo.sf_len+1)]
        episode_return_list = []
        mean_cont_cunt = 0
        episode_state_traj = []
        episode_hybrid_action_traj = []
    
        for i in range(num_eval_episodes):
            episode_return, dt_ratio, trajectory, cont_cunt = self.eval_sched_step(self.env_test, self.ob_test)
            mean_return += episode_return / num_eval_episodes
            mean_cont_cunt += cont_cunt / num_eval_episodes
            # === for state and action exploration check ===
            episode_state_traj.append(trajectory['state'])
            episode_hybrid_action_traj.append(trajectory['hybrid_action'])
            # ===
            episode_return_list.append(episode_return)
            for j in range(len(mean_dt_ratio)):
                mean_dt_ratio[j] += dt_ratio[j]
        # === return std ===
        std_return = np.std(np.array(episode_return_list))  # 단일 시스템
        # === control frequency ===
        print(f"mean_cont_cunt:{mean_cont_cunt}")
        
        # === print discrete action ratio ===
        print("dt ratio:", end='')
        total_action = sum(mean_dt_ratio)
        for i in range(self.algo.sf_len+1):
            print("[{}] {}% | ".format(i, round((mean_dt_ratio[i]/total_action)*100, 1)), end='')
        print("")

        # Log to CSV.
        if not exploration_check:
            self.log["step"].append(step_env)
            self.log["return"].append(mean_return)
            self.log["dt_ratio"].append(mean_dt_ratio)
            self.log["trajectory"].append(0)
            self.log["std_return"].append(std_return)
            self.log["cont_cunt"].append(mean_cont_cunt)
            pd.DataFrame(self.log).to_csv(self.csv_path, index=False)
        
        # ===== Log to CSV with dictionary =====
        for idx, state_traj in enumerate(episode_state_traj):
            if exploration_check:
                np.savetxt(os.path.join(self.log_dir, f'envstep-{step_env}_episode-{idx}_state_log.csv'), state_traj, delimiter=',')
                np.savetxt(os.path.join(self.log_dir, f'envstep-{step_env}_episode-{idx}_hybrid_action_log.csv'), episode_hybrid_action_traj[idx], delimiter=',')
            else:
                np.savetxt(os.path.join(self.log_dir, f'episode-{idx}_state_log.csv'), state_traj, delimiter=',')
                np.savetxt(os.path.join(self.log_dir, f'episode-{idx}_hybrid_action_log.csv'), episode_hybrid_action_traj[idx], delimiter=',')


        # Log to TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step_env)
        print(f"Steps: {step_env:<6}   " f"Return: {mean_return:<5.1f}   " f"Time: {self.time}")
    

    def motivation_evaluate(self, step_env):
        mean_return = 0.0
        mean_dt_ratio = [0 for _ in range(self.algo.sf_len+1)]
        period_sample = [2, 8, 20, 30]
        motivation_return = {}
        mean_cont_cunt = 0
        for p in period_sample:
            motivation_return[p] = 0.

        for p in period_sample:
            for i in range(self.num_eval_episodes):
                episode_return, _, cont_cunt = self.eval_sched_step(self.env_test, self.ob_test, dt_=p)
                motivation_return[p] += episode_return / self.num_eval_episodes
                mean_cont_cunt += cont_cunt / self.num_eval_episodes
                


        # Log to CSV.
        self.log["step"].append(step_env)
        self.log["return"].append(mean_return)
        self.log["dt_ratio"].append(mean_dt_ratio)
        self.log["trajectory"].append(0)
        self.log["motivation_return"].append(motivation_return)
        self.log["std_return"].append(0)
        self.log["cont_cunt"].append(mean_cont_cunt)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)


        # Log to TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step_env)
        print(f"Steps: {step_env:<6}  |  " f"Time: {self.time}  |  ", end='')
        for p in period_sample:
            print(f"Return-period[{p}]: {motivation_return[p]:<5.1f}  |  ", end='')
        print("")