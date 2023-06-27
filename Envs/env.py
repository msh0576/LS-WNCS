import dmc2gym
import gym
import random
gym.logger.set_level(40)


def make_dmc(domain_name, task_name, action_repeat=1, image_size=64, use_image=False):
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        visualize_reward=False,
        from_pixels=True if use_image else False,
        height=image_size,
        width=image_size,
        frame_skip=action_repeat,
    )
    setattr(env, 'action_repeat', action_repeat)
    return env

def delay_step(env, curr_action, prev_action, dt, act_delay, obs_delay):
    acc_reward = 0.

    for tp in range(dt):
        if tp <= act_delay + obs_delay - 1:
            action = prev_action
        else:
            action = curr_action
        # print("tp:{} --- applied action: {}".format(tp, action))
        next_state, reward, done, _ = env.step(action)    # ob: tensor, [1, D_obs] | reward: scalar
        acc_reward += reward
        if done:
            break
    info = {}
    return next_state, acc_reward, done, info

def dt_step(env, action, dt):
    acc_reward = 0.

    for tp in range(dt):
        next_state, reward, done, _ = env.step(action)    # ob: tensor, [1, D_obs] | reward: scalar
        acc_reward += reward
        if done:
            break
    info = {}
    return next_state, acc_reward, done, info

def get_delay(pkt_loss):
    delay = 0
    while True:
        if random.random() < pkt_loss:
            delay += 1
        else:
            break
    return delay



