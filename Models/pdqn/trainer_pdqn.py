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

class SlacObservation:
    """
    Observation for SLAC.
    """

    def __init__(self, state_shape, action_shape, num_sequences):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.num_sequences = num_sequences

    def reset_episode(self, state):
        self._state = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        for _ in range(self.num_sequences - 1):
            self._state.append(np.zeros(self.state_shape, dtype=np.uint8))
            self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        self._state.append(state)

    def append(self, state, action):
        self._state.append(state)
        self._action.append(action)


    @property
    def state(self):
        return np.array(self._state)[None, ...]

    @property
    def action(self):
        return np.array(self._action).reshape(1, -1)


class PdqnTrainer:
    def __init__(
        self,
        env,
        env_test,
        algo,
        log_dir,
        device,
        num_plant=7,
        seed=0,
        delay_env=False,
        num_steps=3 * 10 ** 6,
        initial_collection_steps=10 ** 4,
        initial_learning_steps=10 ** 5,
        num_sequences=8,
        eval_interval=10 ** 4,
        num_eval_episodes=5,
        render=False,
        algo_name='slac',
        initial_episodes=100,
        num_episodes=250,
        eval_interval_epi = 10,
    ):
        # Env to collect samples.
        self.env = env
        self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        self.env_test.seed(2 ** 31 - seed)

        # Observations for training and evaluation.
        self.ob = SlacObservation(
            env.observation_space.shape, 
            env.action_space.shape if algo_name != 'hsac' else (2,), 
            num_sequences
        )
        self.ob_test = SlacObservation(
            env.observation_space.shape, 
            env.action_space.shape if algo_name != 'hsac' else (2,), 
            num_sequences
        )

        # Algorithm to learn.
        self.algo = algo
        self.algo_name = algo_name

        # Log setting.
        self.log = {"step": [], "return": [], "dt_ratio": [], "std_return":[], "cont_cunt":[]}
        self.csv_path = os.path.join(log_dir, "log.csv")
        self.log_dir = log_dir
        self.summary_dir = os.path.join(log_dir, "summary")
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, "model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.action_repeat = self.env.action_repeat
        self.num_steps = num_steps
        self.initial_collection_steps = initial_collection_steps
        self.initial_learning_steps = initial_learning_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.render = render
        self.num_plant = num_plant
        self.delay_env = delay_env
        self.device = device
        self.initial_episodes = initial_episodes
        self.num_episodes = num_episodes
        self.eval_interval_epi = eval_interval_epi
    
    def episode_train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()
        self.algo.prev_state = state
        act, act_param, all_action_parameters = self.algo.random_act()

        terminal = True
        
        # === Collect trajectories using random policy. ===
        print(f"[Initial collection steps]")
        for epi in tqdm(range(1, self.initial_episodes + 1)):
            done = False
            while not done:
                t, _ , done, act, act_param, all_action_parameters = self.algo.step_env(
                    self.env, t,  
                    act, act_param, all_action_parameters, 
                    True,
                    delay_step_=self.delay_env
                )

                if done:
                    t = 0
                    state = self.env.reset()
                    state = np.array(state, dtype=np.float32, copy=False)
                    self.algo.prev_state = state
                    act, act_param, all_action_parameters = self.algo.act(self.algo.prev_state)
                

        # === Iterate collection, update and evaluation. ===

        for epi in tqdm(range(self.initial_episodes + 1, self.num_episodes)):
            done = False

            while not done:
                if done:
                    t = 0
                    state = self.env.reset()
                    state = np.array(state, dtype=np.float32, copy=False)
                    self.algo.prev_state = state
                    act, act_param, all_action_parameters = self.algo.act(self.algo.prev_state)

                t, _ , done, act, act_param, all_action_parameters = self.algo.step_env(
                    self.env, t,  
                    act, act_param, all_action_parameters, 
                    False,
                    delay_step_=self.delay_env
                )

            # === evaluation ===
            if epi % self.eval_interval_epi == 0:
                self.evaluate(epi)
                self.algo.save_models(os.path.join(self.model_dir, f"epi{epi}"))
            
    
    
    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()
        self.algo.prev_state = state
        act, act_param, all_action_parameters = self.algo.random_act()

        terminal = True
        # === Collect trajectories using random policy. ===
        print(f"[Initial collection steps]")
        for step in tqdm(range(1, self.initial_collection_steps + 1)):
            t, _ , terminal, act, act_param, all_action_parameters = self.algo.step_env(
                self.env, t,  
                act, act_param, all_action_parameters, 
                step <= self.initial_collection_steps,
                delay_step_=self.delay_env
            )

            if terminal:
                t = 0
                state = self.env.reset()
                state = np.array(state, dtype=np.float32, copy=False)
                self.algo.prev_state = state
                act, act_param, all_action_parameters = self.algo.act(self.algo.prev_state)

        

        step_env = 0
        # Collect trajectories using random policy.
        # for step in tqdm(range(1, self.num_steps // (self.num_plant+1) + 1)):
        for step in tqdm(range(self.initial_collection_steps + 1, self.num_steps)):
            if terminal:
                t = 0
                state = self.env.reset()
                state = np.array(state, dtype=np.float32, copy=False)
                self.algo.prev_state = state
                act, act_param, all_action_parameters = self.algo.act(self.algo.prev_state)

            t, _ , terminal, act, act_param, all_action_parameters = self.algo.step_env(
                self.env, t,  
                act, act_param, all_action_parameters, 
                step <= self.initial_collection_steps,
                delay_step_=self.delay_env
            )

            # step_env = step * (self.num_plant+1)
            if step % self.eval_interval == 0:
                self.evaluate(step)
                self.algo.save_models(os.path.join(self.model_dir, f"step{step}"))


    
    def evaluate(self, step_env):
        mean_return = 0.0
        mean_dt_ratio = [0 for _ in range(self.algo.num_actions)]
        episode_return_list = []
        mean_cont_cunt = 0
        
        for i in range(self.num_eval_episodes):
            episode_return, dt_ratio, cont_cunt = self.algo.evaluate_steps(self.env_test)

            mean_return += episode_return / self.num_eval_episodes
            mean_cont_cunt += cont_cunt / self.num_eval_episodes
            episode_return_list.append(episode_return)

            for j in range(len(mean_dt_ratio)):
                mean_dt_ratio[j] += dt_ratio[j]
        # === return std ===
        std_return = np.std(np.array(episode_return_list))  # 단일 시스템
        # === control frequency ===
        print(f"mean_cont_cunt:{mean_cont_cunt}")
        
        # print discrete action ratio
        print("dt ratio:", end='')
        total_action = sum(mean_dt_ratio)
        for i in range(self.algo.num_actions):
            print("[{}] {}% | ".format(i, round((mean_dt_ratio[i]/total_action)*100, 1)), end='')
        print("")

        # Log to CSV.
        self.log["step"].append(step_env)
        self.log["return"].append(mean_return)
        self.log["dt_ratio"].append(mean_dt_ratio)
        self.log["std_return"].append(std_return)
        self.log["cont_cunt"].append(mean_cont_cunt)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

        # Log to TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step_env)
        print(f"Steps: {step_env:<6}   " f"Return: {mean_return:<5.1f}   " f"Time: {self.time}")

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))