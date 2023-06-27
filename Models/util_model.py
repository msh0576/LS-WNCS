import numpy as np
from collections import Counter
from copy import deepcopy

class epsilon_decay():
    def __init__(self, eps_decay, eps_final):
        self.eps_decay = eps_decay
        if eps_decay == 0:
            self.eps = eps_final
        else:
            self.eps = 1.
        self.eps_final = eps_final
        self.i_step = 0
    
    def decay_eps(self, syst_id=None):
        if self.eps_decay > 0 and self.eps > self.eps_final:
            self.eps = max(self.eps_final, 1. - self.eps_decay * self.i_step)
            if syst_id is not None:
                if syst_id == 0:
                    self.i_step += 1
            else:
                self.i_step += 1
        return self.eps

def future_schedule_state(state, timer, syst_id, future_len, num_plant):
    '''

    Input:
        state: np, [num_plant, D_state]
        timer: dict,
        syst_id:
        num_plant
    '''
    # print(f"timer:{timer}")
    state = state.reshape(-1)
    curr_time = timer[syst_id]
    future_sched = []
    for idx in range(num_plant):
        if idx != syst_id:
            future_sched.append(timer[idx])

    future_sched = [x for x in future_sched if x > curr_time and x <= curr_time + future_len]
    # === one-hot future_sched ===
    one_hot_future_sched = np.zeros(future_len, dtype=np.float32)
    future_idx = [x - curr_time - 1 for x in future_sched]
    # print(f"curr_time:{curr_time} | future_sched:{future_sched} | future_idx:{future_idx}")
    one_hot_future_sched[future_idx] = 1.
    # print(f"one_hot_future_sched:{one_hot_future_sched}")
    adj_state = np.concatenate([state, one_hot_future_sched], axis=0)
    # print(f"adj_state:{adj_state.shape}")
    return adj_state, one_hot_future_sched

def schedulable_check(timer, syst_id, dt, num_plant, min_period, max_period):
    up_adj_dt = deepcopy(dt)
    up_my_sched_time = timer[syst_id] + up_adj_dt

    # === check whether already scheduled or not ===
    schedulable = False
    while not schedulable:
        schedulable_cunt = 0
        for idx in range(num_plant):
            if timer[idx] == up_my_sched_time:
                up_adj_dt += 1
                up_my_sched_time = timer[syst_id] + up_adj_dt
                break
            else:
                schedulable_cunt += 1
        if schedulable_cunt == num_plant:
            schedulable = True
    # === inverse schedulable check ===
    down_adj_dt = deepcopy(dt)
    down_my_sched_time = timer[syst_id] + down_adj_dt
    schedulable = False
    while not schedulable:
        schedulable_cunt = 0
        for idx in range(num_plant):
            if timer[idx] == down_my_sched_time:
                down_adj_dt -= 1
                down_my_sched_time = timer[syst_id] + down_adj_dt
                break
            else:
                schedulable_cunt += 1
        if schedulable_cunt == num_plant:
            schedulable = True
    
    # === 낮은 주기부터 선택 ===
    if down_adj_dt >= min_period:
        adj_dt = down_adj_dt
    elif down_adj_dt < min_period and up_adj_dt <= max_period:
        adj_dt = up_adj_dt
    else:
        # adj_dt = max_period
        adj_dt = up_adj_dt
        
    # print(f"========timer[{syst_id}]:{timer[syst_id]+dt} | up_adj_dt:{up_adj_dt} | down_adj_dt:{down_adj_dt}")
    # if abs(timer[syst_id] - up_my_sched_time) < abs(timer[syst_id] - down_my_sched_time):
    #     adj_dt = min(up_adj_dt, np.array([max_period]))
    # else:
    #     adj_dt = max(down_adj_dt, np.array([min_period]))

    return adj_dt

def timer_check(timer, num_plant, max_step=1000):
    times = []
    conflict = False
    for syst_id in range(num_plant):
        if timer[syst_id] > 0 and timer[syst_id] < max_step:
            times.append(timer[syst_id])
    
    if len(times) != len(set(times)):
        conflict = True
    return conflict
    
def get_curr_time(timer):
    min_time = 1000000
    for key, time in timer.items():
        if key != 'global':
            if time < min_time:
                min_time = time
    return min_time

def tmp_schedulable_check(timer, syst_id, dt, num_plant, min_period, max_period):
    up_adj_dt = deepcopy(dt)
    up_my_sched_time = timer[syst_id] + up_adj_dt

    # === check whether already scheduled or not ===
    schedulable = False
    while not schedulable:
        schedulable_cunt = 0
        for idx in range(num_plant):
            if timer[idx] == up_my_sched_time:
                up_adj_dt += 1
                up_my_sched_time = timer[syst_id] + up_adj_dt
                break
            else:
                schedulable_cunt += 1
        if schedulable_cunt == num_plant:
            schedulable = True
    # === inverse schedulable check ===
    down_adj_dt = deepcopy(dt)
    down_my_sched_time = timer[syst_id] + down_adj_dt
    schedulable = False
    while not schedulable:
        schedulable_cunt = 0
        for idx in range(num_plant):
            if timer[idx] == down_my_sched_time:
                down_adj_dt -= 1
                down_my_sched_time = timer[syst_id] + down_adj_dt
                break
            else:
                schedulable_cunt += 1
        if schedulable_cunt == num_plant:
            schedulable = True
    # print(f"========timer[{syst_id}]:{timer[syst_id]+dt} | up_adj_dt:{up_adj_dt} | down_adj_dt:{down_adj_dt}")
    if abs(timer[syst_id] - up_my_sched_time) < abs(timer[syst_id] - down_my_sched_time):
        adj_dt = min(up_adj_dt, np.array([max_period]))
    else:
        adj_dt = max(down_adj_dt, np.array([min_period]))

    return adj_dt



def scheduling(elap_time, dts, num_plant, sf_len):
    '''
    Input:
        elap_time: np, [D_plant,]
        dts: np, [D_plant,]
    '''
    sched = np.ones((sf_len,), dtype=np.int32) * num_plant
    empty_val = num_plant
    elap_time_ = deepcopy(elap_time)
    syst_priority = elap_time_.argsort()
    high_order_syst = np.argsort(syst_priority)[::-1]
    # print(f"elap_time_:{elap_time_} | syst_priority:{syst_priority} | high_order_syst:{high_order_syst}")

    for s in high_order_syst:
        for idx, val in enumerate(sched):
            if val == empty_val and elap_time_[s] >= dts[s]:
                sched[idx] = s
                elap_time_[s] = 1
            else:
                elap_time_[s] += 1
    
    return sched



def select_syst(timer):
    '''
    slect a system with a minimum timer ()
    Input:
        timer: {
            'global':
            0:
            1:
            ...
        }
    '''
    minimum = 10000000
    syst_id = None
    for key, time in timer.items():
        if key == 'global':
            pass
        else:
            if time < minimum:
                minimum = time
                syst_id = key
    return syst_id

def postprocess_sf_sched(sf_sched, num_plant, sf_len, min_period):
    '''
    sf_sched 에서, 각 시스템에 할당된 주기가 최소값을 만족하도록 수정
    Input:
        sf_sched: np, (sf_len, ): 
    Output:
        proc_sf_sched = {
            '0': {'position', 'period},
            '1': {'position', 'period},
            ...
        }
    '''
    proc_sf_sched = {}
    syst_id = 0
    for syst_id in range(num_plant):
        while_cunt = 0
        while while_cunt < sf_len+1:
            while_cunt += 1
            position = np.where(sf_sched==syst_id)[0]
            period = position[1:] - position[:-1]
            for idx, val in enumerate(period):
                if val < min_period:
                    sf_sched[position[idx+1]] = num_plant
                    break
                if idx == len(period) - 1:
                    while_cunt = sf_len+1
        proc_sf_sched[syst_id] = {'position': position, 'period': period}

    return proc_sf_sched
    
def dts_adjustment(dts, sf_len):
    '''
    Ex:
        dts = [5, 8, 10, 12, 6], then 
        adj_dts = [5, 8, 10, 12, 0]
    '''
    adj_dts = deepcopy(dts)
    N, L = dts.shape

    for syst_idx, dts_ in enumerate(dts):
        sched_len = 0
        for idx, dt in enumerate(dts_):
            sched_len += dt
            if sched_len >= sf_len and idx < L-1:
                adj_dts[syst_idx, idx+1:] = 0
    return adj_dts

def sf_to_k(sf_sched, min_, max_):
    '''
    Input:
        sf_sched: np, [sf_len], ex: [0, 0, 1, 1, 2, 0, 2, ...]
    '''
    k = k_normal(sf_sched, min_, max_)
    return k

def k_to_sf(k, min_, max_):
    return np.array(list(map(lambda x: k_unnormal(*x), (k, min_, max_))), dtype=np.int32)

def k_to_dts(k, min_, max_):
    if len(k.shape) == 1:
        N = k.shape[0]
        min_ = np.repeat(min_, N)
        max_ = np.repeat(max_, N)
        result = np.array(list(map(lambda x, y, z: k_unnormal(x, y, z), k, min_, max_)), dtype=np.int32)
    else:
        N = k.shape[1]
        min_ = np.repeat(min_, N)
        max_ = np.repeat(max_, N)
        tmp = []
        for k_ in k:
            tmp.append(list(map(lambda x, y, z: k_unnormal(x, y, z), k_, min_, max_)))
        result = np.stack(tmp, axis=0).astype(np.int32)

    return result

def tmp_dts_to_k(dts, min_, max_):
    dts_ = deepcopy(dts)
    z = np.where(dts_ < min_)
    k = k_normal(dts, min_, max_)
    k[z] = 0
    return k

def dts_to_k(dts, min_, max_):
    k = k_normal(dts, min_, max_)
    return k

def count_dt_ratio(total_sched, num_plant, sf_len):
    total_sched = np.array(total_sched, dtype=np.int32)
    syst_dt_ratio = {}
    for syst_id in range(num_plant):
        syst_dt_ratio[syst_id] = [0 for _ in range(sf_len)]

        position = np.where(total_sched==syst_id)[0]
        periods = Counter(position[1:] - position[:-1])
        # print(f"period:{periods}")

        for key, period in periods.items():
            if key <= sf_len:
                syst_dt_ratio[syst_id][key-1] = period
    # print(f"syst_dt_ratio:{syst_dt_ratio}")
    return syst_dt_ratio

def k_unnormal(k_norm, min_, max_):
    return round( ((k_norm+1) / 2) * (max_ - min_) + min_)

def k_normal(k, min_, max_):
    return np.array(2 * (k - min_) / (max_ - min_) - 1, dtype=np.float32)