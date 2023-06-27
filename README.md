# Learning-Enabled Network-Control Co-Design for Energy-Efficient Industrial Internet of Things

This is the repository for our paper "Learning-Enabled Network-Control Co-Design for Energy-Efficient Industrial Internet of Things"

# Requirements
- Python 3.8.3
- PyTorch (v1.7.1)
- mujoco_py (v2.0)
- gym (v0.22)

# Instructions

To train the LS-WNCS module:
```
python3 run_LS_WNCS.py --domain_name cartpole --task_name swingup --algo_name lswncs --cuda_id 3 --condition_name test1  --multi_env --seed 2 --episode_train --initial_episodes 3000 --num_episodes 6000 --multipolicy
```

To train the baseline algorithms:
```
python3 run_LS_WNCS.py --domain_name cartpole --task_name swingup --algo_name hsac --cuda_id 2 --condition_name test1  --encoder_type identity_robust --decoder_type identity_robust  --seed 2 --episode_train --initial_episodes 3000 --num_episodes 6000 --multipolicy
```

```
python3 run_LS_WNCS.py --domain_name cartpole --task_name swingup --algo_name mpdqn --cuda_id 3 --condition_name test1  --encoder_type identity_robust --decoder_type identity_robust  --seed 2 --episode_train --initial_episodes 3000 --num_episodes 6000 --multipolicy
```

To show figures:
```
python3 figures.py
```

