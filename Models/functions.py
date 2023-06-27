import random
import numpy as np

class scheduler():
    def __init__(self, algo, schedule_len, chunk_size=1, max_period=20):
        candi_algo = ['sequential', 'random', 'chunk']
        assert algo in candi_algo
        if algo == 'sequential':
            self.algo = _sequentialSchedule(schedule_len)
        elif algo == 'random':
            self.algo = _randomSchedule(schedule_len, max_period)
        elif algo == 'chunk':
            self.algo = _chunkSchedule(schedule_len, chunk_size)
    
    def get_schedule(self):
        return self.algo.get_schedule()
    
    def get_period(self, syst_idx):
        '''
        Output:
            period: scalar
        '''
        return self.algo.get_period(syst_idx)
    
    def choice_period(self, cand_periods, weights, sample=1):
        return self.algo.choice_period(cand_periods, weights, sample=1)
    

class _sequentialSchedule():
    def __init__(self, schedule_len):
        self.sched_idx = 0
        self.schedule_len = schedule_len
    
    def get_schedule(self):
        '''
            Input:
                schedule_size: possible schedule range
            Output:
                # sched: np, [1, ], current scheduled system index
                sched: scalar, current scheduled system index
        '''
        sched = np.array([self.sched_idx % self.schedule_len], dtype=np.float32)
        self.sched_idx += 1
        return sched.item(0)
    
    def get_period(self, syst_idx):
        '''
        Input:
            syst_idx: scalar, the system that will get the period
        Output:
            period: scalar, current exist system index
        '''
        periods = np.ones(shape=(self.schedule_len-1,), dtype=np.float32) * self.schedule_len

        return periods[syst_idx]
    
    def choice_period(self, cand_periods, weights, sample=1):
        '''
        Choose a priod in the candidate periods with the probabilities
        Input:
            cand_periods: list
            probs: list, sum = 1.
        Output:
            result: list
        '''
        assert len(cand_periods) == len(weights)
        assert sum(weights) == 1.
        return random.choices(cand_periods, weights, k=sample)[0]


class _randomSchedule():
    def __init__(self, schedule_len, max_period=20):
        self.schedule_len = schedule_len
        self.max_period = max_period

    def get_schedule(self):
        '''
        Input:
            schedule_len: 
        Output:
            sched: np, [1, ]
        '''
        sched = random.randint(0, self.schedule_len)
        sched = np.array([sched], dtype=np.float32)
        return sched
    
    def get_period(self, syst_idx):
        '''
        Input:
            syst_idx: scalar, the system that will get the period
        Output:
            period: scalar, period range [1 ~ max_period]
        '''
        periods = [random.randint(1, self.max_period) for _ in range(self.schedule_len-1)]

        return periods[syst_idx]

class _chunkSchedule():
    def __init__(self, num_plant, chunk_size):
        self.num_plant = num_plant
        self.chunk_size = chunk_size
        self.global_sched_len = num_plant * chunk_size
        self.chunk_idx = 0
        self.chunk_cunt = 0
    
    def get_schedule(self):
        '''
        Output:
            sched: np, [1,]
        '''
        sched = np.array([self.chunk_idx], dtype=np.float32)
        self.chunk_cunt += 1
        if self.chunk_cunt == self.chunk_size:
            self.chunk_idx = (self.chunk_idx + 1) % self.num_plant
            self.chunk_cunt = 0
        return sched


def softmax(a) : 
    c = np.max(a) # 최댓값
    exp_a = np.exp(a-c) # 각각의 원소에 최댓값을 뺀 값에 exp를 취한다. (이를 통해 overflow 방지)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def is_sorted(a):
    for i in range(a.size-1):
         if a[i+1] < a[i] :
               return False
    return True