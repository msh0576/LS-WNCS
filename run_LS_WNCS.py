

import os
import argparse
import random
import time
import numpy as np
import torch

def set_arguments():
    parser = argparse.ArgumentParser('Running model-based RL')
    #--- mpn ---#
    parser.add_argument('--action_repeat', type=int, default=1, metavar='R', help='Action repeat')
    parser.add_argument('--num_plant', type=int, default=1, help='number of plants')
    parser.add_argument('--cuda_id', type=int, default=0, help='CUDA id to use')
    parser.add_argument('--app-final-idx', type=int, default=1, help='Final idx of MPN application')
    parser.add_argument('--app-start-idx', type=int, default=1, help='Start idx of MPN application')
    parser.add_argument('--pkt_loss', type=float, default=0., help='packet loss probability')

    # === 
    parser.add_argument("--domain_name", type=str, default="cheetah")
    parser.add_argument("--task_name", type=str, default="run")
    parser.add_argument("--num_steps", type=int, default=250000)
    parser.add_argument("--render", action='store_true', help='render')
    parser.add_argument("--algo_name", type=str, default='slac', 
        choices=['lswncs', 'mpdqn', 'hsac'], 
        help='Implemented candidate algorithms'
    )
    # parser.add_argument("--use_image", action='store_true', help='use image state')
    parser.add_argument("--condition_name", type=str, default='none', help='for log_dir, add experiment conditions such as loss probabilities, etc')
    parser.add_argument("--initial_learning_steps", type=int, default=1000, help='initial learning steps')
    parser.add_argument("--delay_env", action='store_true', help='Use delayed environment')

    # === 
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--obs_normal', action='store_true')
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
    parser.add_argument('--num_sequences', default=1, type=int)
    # train
    parser.add_argument('--agent', default='sac_ae', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-4, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-4, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='identity_robust', choices=['pixel', 'identity', 'identity_robust'], type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--decoder_type', default='identity_robust', choices=['pixel', 'identity', 'identity_robust'], type=str)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float)
    parser.add_argument('--decoder_weight_lambda', default=1e-7, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    # ===
    parser.add_argument('--tau', default=0.005, type=float)	            # actor network size
    parser.add_argument('--vae_itr', default=50000, type=int)		    # vae training iterations
    parser.add_argument('--max_period', default=30, type=int)
    parser.add_argument('--min_period', default=1, type=int)
    parser.add_argument('--sf_len', default=30, type=int)
    parser.add_argument('--initial_collection_steps', default=100000, type=int)
    parser.add_argument('--latent_dim', default=30, type=int)
    parser.add_argument('--sched_method', default='slot', choices=['sf', 'slot'], type=str)
    parser.add_argument('--multi_env', default=False, action='store_true')
    parser.add_argument('--embed_size', default=0, type=int)
    parser.add_argument('--nstep_return', default=False, action='store_true')
    parser.add_argument('--episode_train', default=False, action='store_true')
    parser.add_argument('--initial_episodes', default=3000, type=int, help='the number of episodes on VAE training (offline)')
    parser.add_argument('--num_episodes', default=6000, type=int, help='the number of episodes on agent model training (online)')
    parser.add_argument('--multipolicy', default=False, action='store_true')
    parser.add_argument('--loss_energy', default=False, action='store_true', help='Consider energy loss term in VAE model')
    parser.add_argument('--energy_coeff', default=1., type=float, help='energy loss coefficienty in VAE model')

    args = parser.parse_args()
    return args

from Envs.env import make_dmc
from Utils import utils
from Models.ls_wncs_model import LSWNCSAlgorithm
from Trainer.trainer_lswncs import TrainerLSWNCS

from Models.pdqn.algo_mpdqn import MultiPassPDQNAgent
from Models.pdqn.trainer_pdqn import PdqnTrainer

from Models.sac.trainer_sac import TrainerSac
from Models.hybrid_sac.algo_hsac import HsacAlgorithm
from Models.sac.algo_sac import SacAlgorithm

from datetime import datetime
import pytz


def main_project(args, app_idx=1):
    assert args.encoder_type == args.decoder_type
    

    if args.multi_env:
        env_list = [
            make_dmc(
                domain_name=args.domain_name,
                task_name=args.task_name,
                action_repeat=args.action_repeat,
                image_size=84 if args.algo_name == 'sac_ae' else 64,
                use_image=(args.encoder_type == 'pixel')
            ) for _ in range(args.num_plant)
        ]
        env_test_list = [
            make_dmc(
                domain_name=args.domain_name,
                task_name=args.task_name,
                action_repeat=args.action_repeat,
                image_size=84 if args.algo_name == 'sac_ae' else 64,
                use_image=(args.encoder_type == 'pixel')
            ) for _ in range(args.num_plant)
        ]
    else:
        env = make_dmc(
            domain_name=args.domain_name,
            task_name=args.task_name,
            action_repeat=args.action_repeat,
            image_size=84 if args.algo_name == 'sac_ae' else 64,
            use_image=(args.encoder_type == 'pixel')
        )
        env_test = make_dmc(
            domain_name=args.domain_name,
            task_name=args.task_name,
            action_repeat=args.action_repeat,
            image_size=84 if args.algo_name == 'sac_ae' else 64,
            use_image=(args.encoder_type == 'pixel')
        )
        

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)
        env_test = utils.FrameStack(env_test, k=args.frame_stack)


    process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))

    log_dir = os.path.join(
        "logs",
        f"{args.domain_name}-{args.task_name}",
        process_start_time.strftime("%Y%m%d_%H%M%S") + \
        f'_{args.algo_name}-{args.condition_name}-seed{args.seed}',
    )
    if not os.path.exists(log_dir):
        utils.make_dir(log_dir)

    print("log_dir:", log_dir)
    print(f"args:{vars(args)}")
    print("================================ \n\n")


    

    device = torch.device('cuda:{}'.format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    if 'lswncs' in args.algo_name:
        algo_fn = LSWNCSAlgorithm
        env = env_list[0]
        algo = algo_fn(
            state_shape=env.observation_space,
            action_shape=env.action_space,
            schedule_size=2,    # number of schdulable system
            # max_period=args.max_period,
            # sf_len=args.sf_len,
            # latent_dim=40,
            device=device,
            sf_sched=args.sched_method=='sf',
            **vars(args),
        )
    elif 'mpdqn' in args.algo_name:
        pdqn_fn = MultiPassPDQNAgent
        algo = pdqn_fn(
            env.observation_space, env.action_space,
            schedule_size=args.max_period,   # maximum period
            sf_len=1,
            actor_kwargs={'hidden_layers': [128],
                            'action_input_layer': 0,},
            actor_param_kwargs={'hidden_layers': [128],
                                'squashing_function': False,
                                'output_layer_init_std': 0.0001,},
            device=device,
            seed=args.seed,
            algo_name=args.algo_name,
            weighted=True,
        )
    elif args.algo_name == 'hsac':
        algo = HsacAlgorithm(
            buffer_size=10000, 
            batch_size=128, 
            input_shape=env.state_space.shape[0], 
            out_c=env.action_space.shape[0], 
            out_d=30, 
            device=device,
            num_plant=args.num_plant, 
            obs_normal=args.obs_normal,
            num_sequences=args.num_sequences,
            sf_sched=args.sched_method=='sf',
        )
    elif args.algo_name == 'sac':
        algo = SacAlgorithm(
            buffer_size=10000,
            batch_size=128,
            input_shape=env.observation_space.shape[0],
            out_c=env.action_space.shape[0],
            device=device,
            pkt_loss=args.pkt_loss,
            sf_len=args.sf_len,
            max_period=args.max_period,
            num_plant=args.num_plant,
        )

    if args.algo_name in ['lswncs', 'sac', 'hsac']:
        if 'lswncs' in args.algo_name:
            trainer_fn = TrainerLSWNCS
        elif args.algo_name == 'sac' or args.algo_name == 'hsac':
            trainer_fn = TrainerSac
        else:
            raise Exception("algo_name error!")
        trainer = trainer_fn(
            env=env if not args.multi_env else env_list,
            env_test=env_test if not args.multi_env else env_test_list,
            algo=algo,
            log_dir=log_dir,
            seed=args.seed,
            num_steps=args.num_steps,
            device=device,
            render=args.render,
            algo_name=args.algo_name,
            num_sequences=8 if args.algo_name in ['slac', 'hsac'] else args.num_sequences,
            delay_env=args.delay_env,
            num_plant=args.num_plant,
            initial_collection_steps=args.initial_collection_steps,
            # initial_learning_steps=10 ** 5,
            initial_learning_steps=args.initial_learning_steps,
            vae_itr=args.vae_itr,
            sf_sched=args.sched_method=='sf',
            embed_size=args.embed_size,
            eval_interval=10**3,    # 10**3
            initial_episodes=args.initial_episodes,
            num_episodes=args.num_episodes,
            eval_interval_epi=50,   # origin:50
        )
    elif 'pdqn' in args.algo_name:
        trainer = PdqnTrainer(
            env=env,
            env_test=env_test,
            algo=algo,
            log_dir=log_dir,
            seed=args.seed,
            num_steps=args.num_steps,
            device=device,
            eval_interval=10 ** 3,
            # initial_learning_steps=10 ** 5,
            initial_learning_steps=args.initial_learning_steps,
            render=args.render,
            algo_name=args.algo_name,
            num_sequences=8 if args.algo_name in ['slac', 'hsac'] else 1,
            delay_env=args.delay_env,
            num_plant=args.num_plant,
            initial_collection_steps=args.initial_collection_steps,
            initial_episodes=args.initial_episodes,
            num_episodes=args.num_episodes,
            eval_interval_epi=50,   # 50
        )
    
    
    if 'lswncs' in args.algo_name and args.episode_train:
        trainer.episode_train()
    elif 'pdqn' in args.algo_name and args.episode_train:
        trainer.episode_train()
    elif 'sac' in args.algo_name and args.episode_train:
        trainer.episode_train()
    else:
        raise Exception("train error!")
    

if __name__ == '__main__':
    args = set_arguments()
    for app_idx in range(args.app_start_idx, args.app_final_idx+1):
        # run_mpn(args, app_idx=app_idx)
        main_project(args, app_idx=app_idx)

