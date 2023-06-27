environment: dreamer

[LS-WNCS]
python3 run_LS_WNCS.py --domain_name cartpole --task_name swingup --algo_name lswncs --cuda_id 3 --condition_name test1  --multi_env --seed 2 --episode_train --initial_episodes 3000 --num_episodes 6000 --multipolicy

[Hybrid-SAC]
python3 run_LS_WNCS.py --domain_name cartpole --task_name swingup --algo_name hsac --cuda_id 2 --condition_name test1  --encoder_type identity_robust --decoder_type identity_robust  --seed 2 --episode_train --initial_episodes 3000 --num_episodes 6000 --multipolicy

[MPDQN]
python3 run_LS_WNCS.py --domain_name cartpole --task_name swingup --algo_name mpdqn --cuda_id 3 --condition_name test1  --encoder_type identity_robust --decoder_type identity_robust  --seed 2 --episode_train --initial_episodes 3000 --num_episodes 6000 --multipolicy
