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
from gym import wrappers



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


class Trainer:
    """
    Trainer for SLAC.
    """

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
        eval_interval=10 ** 3,  # 10**4
        num_eval_episodes=10,
        render=False,
        algo_name='slac',
        **kwargs
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
        self.log = {"step": [], "return": [], "dt_ratio":[], "trajectory":[]}
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
        #
        # self.env = wrappers.Monitor(self.env, self.log_dir, video_callable=lambda episode_id: True, force=True)


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

        if self.algo_name == 'slac' or self.algo_name == 'hsac':
            self.algo.buffer.reset_episode(state)


        # Collect trajectories using random policy.
        # print("sepc:", self.env.spec)
        # print("env max epi:", self.env.spec.max_episode_steps)
        print(f"[Initial collection steps]")
        for step in tqdm(range(1, self.initial_collection_steps + 1)):
            t, _ , _ = self.algo.step(self.env, self.ob, t, step <= self.initial_collection_steps, delay_step_=self.delay_env)

        # Update latent variable model first so that SLAC can learn well using (learned) latent dynamics.
        # if self.algo_name == 'slac' and self.algo.encoder_type == 'pixel':
        if self.algo_name in ['slac', 'hyar']:
            bar = tqdm(range(self.initial_learning_steps))
            for _ in bar:
                bar.set_description("[Updating latent variable model.]")
                self.algo.update_latent(self.writer)

        # Iterate collection, update and evaluation.
        episode_reward = 0
        episode_iter = 0

        # for step in tqdm(range(self.initial_collection_steps + 1, self.num_steps // self.action_repeat + 1)):
        for step in tqdm(range(self.initial_collection_steps + 1, self.num_steps // (self.num_plant+1) + 1)):
            # === environment steps ===
            t, reward, done = self.algo.step(self.env, self.ob, t, False, delay_step_=self.delay_env)
            episode_reward += reward

            if done:
                # print(f"episode_reward:{episode_reward}")
                episode_reward = 0
                episode_iter += 1
                if self.algo_name == 'hyar' and episode_iter % 10 == 0:
                    for _ in range(100):
                        self.algo.update_latent(self.writer)


            # === Update the algorithm. ===
            if self.algo_name == 'slac':
                # if self.algo.encoder_type == 'pixel':
                self.algo.update_latent(self.writer)
                self.algo.update_sac(self.writer)
            elif self.algo_name == 'sac':
                self.algo.train(step, self.writer)
            elif self.algo_name == 'hsac':
                # self.algo.train(step)
                self.algo.train_step(step)
            elif self.algo_name == 'hyar':
                self.algo.update_policy(step, self.writer)

            # Evaluate regularly.
            # step_env = step * self.action_repeat
            step_env = step * (self.num_plant+1)
            if step % self.eval_interval == 0:
                # print('evaluate!!', step_env, self.eval_interval)
                self.evaluate(step)
                self.algo.save_model(os.path.join(self.model_dir, f"step{step}"))
        # Wait for logging to be finished.
        sleep(5)

    def evaluate(self, step_env):
        mean_return = 0.0
        mean_dt_ratio = [0 for _ in range(self.algo.schedule_size)]

        for i in range(self.num_eval_episodes):
            # state = self.env_test.reset()
            # self.ob_test.reset_episode(state)
            # episode_return = 0.0
            # done = False

            # while not done:
            #     action = self.algo.exploit(self.ob_test)
            #     state, reward, done, _ = self.env_test.step(action)
            #     self.ob_test.append(state, action)
            #     episode_return += reward
            episode_return, dt_ratio = self.algo.evaluate_steps(self.env_test, self.ob_test, delay_step_=self.delay_env)

            mean_return += episode_return / self.num_eval_episodes

            for j in range(len(mean_dt_ratio)):
                mean_dt_ratio[j] += dt_ratio[j]

        # === print discrete action ratio ===
        print("dt ratio:", end='')
        total_action = sum(mean_dt_ratio)
        for i in range(self.algo.schedule_size):
            print("[{}] {}% | ".format(i, round((mean_dt_ratio[i]/total_action)*100, 1)), end='')
        print("")

        # Log to CSV.
        self.log["step"].append(step_env)
        self.log["return"].append(mean_return)
        self.log["dt_ratio"].append(mean_dt_ratio)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

        # Log to TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step_env)
        print(f"Steps: {step_env:<6}   " f"Return: {mean_return:<5.1f}   " f"Time: {self.time}")

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
