import numpy as np

class KalmanFilter():

    def __init__(self, Q = 0.25, R = 11):
        self.Q_init = Q
        self.R_init = R
        self.reset()

    def reset(self):
        self.Q = self.Q_init
        self.R = self.R_init
        self.posteri = 0.0
        self.posteri_error = 1.0

    def estimate(self, measurement = None):
        if measurement is not None: 
            priori = self.posteri
            priori_error = self.posteri_error + self.Q
            K = priori_error / (priori_error + self.R)
            self.posteri = priori + K * (measurement - priori)
            self.posteri_error = (1 - K) * priori_error
        return self.posteri


class NoneFilter():

    def __init__(self):
        pass
        
    def reset(self):
        self.posteri = 0.0

    def estimate(self, measurement = None):
        if measurement is not None: 
            self.posteri = measurement
        return self.posteri


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size): #size is 1000000
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)  # obs_dim=32, 当前时刻的状态
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)  # 下一个时刻的状态 1e6*32
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)  # 1e6*3
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)  # 1e6*1
        self.done_buf = np.zeros([size, 1], dtype=np.float32)  # 1e6*1
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size  # self.max_size=1e6
        self.size = min(self.size+1, self.max_size)  # 这里是当前buffer里存了数据的多少，当一开始数据小于max_size时，ptr与size是一起变化的
        # buffer在添加到size之后会根据ptr的索引从0开始替换之前的数据！！！

    def sample_batch(self, batch_size=32): #输入512，本来是32改成512
        idxs = np.random.randint(0, self.size, size=batch_size)  # 产生随机整数做索引
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class ProcessNoise:
    """
    Generator of Ornstein–Uhlenbeck Process Noise
    """
    def __init__(self, dim):
        self.x = np.zeros((1,dim))
        self.dim = dim
        self.t = 0.1 # time step between adjacent call.
        self.n = 10 # iterations for one call.
        self.dt = self.t/self.n
        self.theta = 1
        self.sigma = np.sqrt(2)

    def next(self):
        for _ in range(self.n):
            self.x = self.x - self.theta * self.x * self.dt + np.random.normal(0, self.sigma,[1,self.dim]) * np.sqrt(self.dt)
        return self.x