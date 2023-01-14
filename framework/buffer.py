import torch
import numpy as np
import copy, glob
from torch.utils.data import Dataset
# from .utils import padding_obs
# from .feature_translation import translate_local_obs
from tqdm import tqdm
import time
from einops import rearrange, repeat

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)): # good method to count in an anti-direction
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

class Buffer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.max_ep_len = 10000
        self.state_dim = config.obs_space
        self.action_dim = config.action_space
        self.length = config.context_len
        self.buffer_size = config.buffer_capacity

        self.buffer_dict = {}
        self.property_list = ['states', 'states_next', 'rewards', 'dones', 'actions']
        self.buffer_dict_clear()

    def buffer_dict_clear(self):
        for item in self.property_list:
            self.buffer_dict[item] = list()

    def insert(self, item_name: str, data: np.ndarray):
        self.buffer_dict[item_name].append(data)
        self.buffer_dict[item_name][1:self.buffer_size]

    def sample(self, batch_size=256, length=30):
        for item in self.property_list:
            if 'state' in item:
                self.buffer_dict[item] = np.array(self.buffer_dict[item]).transpose(1, 0, 2, 3)
            elif 'rewards' in item or 'dones' in item:
                self.buffer_dict[item] = np.array(self.buffer_dict[item]).transpose(1, 0)
            else:
                self.buffer_dict[item] = np.array(self.buffer_dict[item]).transpose(1, 0, 2)
        state = self.buffer_dict['states']
        action = self.buffer_dict['actions']
        reward = self.buffer_dict['rewards']
        done = self.buffer_dict['dones']
        state_next = self.buffer_dict['states_next']
        return state, action, reward, done, state_next


    def preprocess_data(self):
        #   timestep * dim
        len_ls = [item['actions'].shape[0] for item in self.data]
        total_len = np.max(len_ls) + self.length - 1
        self.max_traj_len = total_len

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for idx in range(len(self.data)):
            traj = self.data[idx]
            traj_len = len_ls[idx]

            # s, a, r , done, rtg, timestep, mask
            # for the first padding
            # 0 ,0,0, false, final_rtg_padding, false

            # get sequences from dataset
            s.append(traj['observations'].reshape(1, -1, self.state_dim))
            a.append(traj['actions'].reshape(1, -1, self.action_dim))
            r.append(traj['rewards'].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'].reshape(1, -1))
            elif 'terminal' in traj:
                d.append(np.array(traj['terminal']).reshape(1, -1))
            else:
                d.append(traj['dones'].reshape(1, -1))
            timesteps.append(np.arange(traj_len).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            timesteps[-1][timesteps[-1] <= 0] = 0  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'], gamma=1.).reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([np.zeros((1, 1, 1)), rtg[-1]], axis=1)

            # TODO: padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, total_len - tlen, self.state_dim)), s[-1]], axis=1)
            # s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, total_len - tlen, self.action_dim)) * -0., a[-1]], axis=1) #TODO: why? *-10?
            r[-1] = np.concatenate([np.zeros((1, total_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, total_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, total_len - tlen, 1)), rtg[-1]], axis=1)  # / scale
            timesteps[-1] = np.concatenate([np.zeros((1, total_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, total_len - tlen)), np.ones((1, tlen))], axis=1))
        # print(f"{time.time()}==={batch_idx[0]}", time.time()-start_time)
        self.traj_len = len_ls
        self.state = np.concatenate(s,axis=0)
        self.action = np.concatenate(a,axis=0)
        self.reward = np.concatenate(r,axis=0)
        self.done = np.concatenate(d,axis=0)
        self.rtg = np.concatenate(rtg, axis=0)
        self.timestep = np.concatenate(timesteps, axis=0)
        self.mask = np.concatenate(mask,axis=0)
        ## for cuda mode
        self.state, self.action, self.reward, self.done, self.rtg, self.timesteps, self.mask = self.move_data_to_device(self.state, self.action, self.reward, self.done, self.rtg, self.timestep, self.mask)

    def sample_prepared_data(self, batch_size=256, length=30):
        start_time = time.time()
        # replace set
        replace = False if len(self.data) > batch_size else True
        batch_idx = np.random.choice(np.arange(len(self.data)), size=batch_size, p=self.p, replace=replace)

        mask = np.zeros((len(self.data), self.max_traj_len), dtype=bool)
        # TODO: fix bug if length is short like 30, how can we sample 60 from it?
        for idx in batch_idx:
            while True:
                start_idx = np.random.randint(0, self.traj_len[idx])
                end_idx = start_idx + length
                # since mask is boolean, we should make all span don't cover each other
                if sum(mask[idx, start_idx:end_idx]) ==0:
                    break
            mask[idx, start_idx:end_idx] = True
        #print(f"{time.time()}3==={batch_idx[0]}", time.time() - start_time) # 0.0173
        s = self.state[mask].reshape(batch_size, length, -1)
        #print(f"{time.time()}2==={batch_idx[0]}", time.time() - start_time) # 0.0231
        a = self.action[mask].reshape(batch_size, length, -1)
        r = self.reward[mask].reshape(batch_size, length, -1)
        d = self.done[mask].reshape(batch_size, length, -1)
        #print(f"{time.time()}1==={batch_idx[0]}", time.time() - start_time) # 0.0353
        rtg = np.array(0)#self.rtg[mask].reshape(batch_size, length, -1)
        timesteps = self.timestep[mask].reshape(batch_size, length, -1)
        mask = self.mask[mask].reshape(batch_size, length, -1)
        #print(f"{time.time()}==={batch_idx[0]}", time.time()-start_time) #0.0370
        return s, a, r, d, rtg, timesteps, mask

    def sample_prepared_data_cuda(self, batch_size=256, length=30):
        start_time = time.time()
        # replace set
        replace = False if len(self.data) > batch_size else True
        batch_idx = np.random.choice(np.arange(len(self.data)), size=batch_size, p=self.p, replace=replace)

        mask = np.zeros((len(self.data), self.max_traj_len), dtype=bool)
        # TODO: fix bug if length is short like 30, how can we sample 60 from it?
        for idx in batch_idx:
            while True:
                start_idx = np.random.randint(0, self.traj_len[idx])
                end_idx = start_idx + length
                # since mask is boolean, we should make all span don't cover each other
                if sum(mask[idx, start_idx:end_idx]) ==0:
                    break
            mask[idx, start_idx:end_idx] = True
        mask = torch.from_numpy(mask).to(device=self.device)
        #print(f"{time.time()}3==={batch_idx[0]}", time.time() - start_time) # 0.05
        s = self.state[mask].reshape(batch_size, length, -1)
        a = self.action[mask].reshape(batch_size, length, -1)
        #print(f"{time.time()}2==={batch_idx[0]}", time.time() - start_time) #0.20
        r = self.reward[mask].reshape(batch_size, length, -1)
        d = self.done[mask].reshape(batch_size, length, -1)
        mask = self.mask[mask].reshape(batch_size, length, -1)
        #TODO: fix the implementation
        rtg = np.array(0)#self.rtg[mask].reshape(batch_size, length, -1)
        timesteps = np.array(0)#self.timestep[mask].reshape(batch_size, length, -1)

        #print(f"{time.time()}1==={batch_idx[0]}", time.time()-start_time) #0.24
        return s, a, r, d, rtg, timesteps, mask

    def context_eval_sample(self, idx=256, length=30):
        length = self.traj_len[idx]
        mask = np.zeros((length , self.max_traj_len), dtype=bool)
        for i in range(length):
            mask[i,i:i+self.length] = True

        s = repeat(self.state[idx,:], 't d -> b t d', b = length)
        s = s[mask].reshape(-1, self.length, s.shape[2])

        a = repeat(self.action[idx, :], 't d -> b t d', b=length)
        a = a[mask].reshape(-1, self.length, a.shape[2])

        r = repeat(self.reward[idx, :], 't d -> b t d', b=length)
        r = r[mask].reshape(-1, self.length, r.shape[2])

        # TODO: fix the implementation
        d = np.array(0)  # self.done[mask].reshape(batch_size, length, -1)
        mask = np.array(0)  # self.mask[mask].reshape(batch_size, length, -1)
        rtg = np.array(0)  # self.rtg[mask].reshape(batch_size, length, -1)
        timesteps = np.array(0)  # self.timestep[mask].reshape(batch_size, length, -1)

        return  s, a, r, d, rtg, timesteps, mask

    def sample_data(self, batch_size=256, length=30):
        return self.sample_prepared_data_cuda(batch_size, length)#self.sample_start_data(batch_size, length)

    def sample_start_data(self, batch_size=256, length=30):
        start_time = time.time()
        np.random.seed()
        batch_idx = np.random.choice(np.arange(len(self.data)), size=batch_size, p=self.p)
        #print(f"{time.time()}==start:{batch_idx[0]}")
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for idx in batch_idx:
            traj = self.data[idx]
            end_idx = np.random.randint(0, len(traj['observations']) - 1)+1
            start_idx = end_idx-length if end_idx-length>0 else 0
            # get sequences from dataset
            s.append(traj['observations'][start_idx:end_idx].reshape(1, -1, self.state_dim))
            a.append(traj['actions'][start_idx: end_idx].reshape(1, -1, self.action_dim))
            r.append(traj['rewards'][start_idx: end_idx].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][start_idx: end_idx].reshape(1, -1))
            else:
                d.append(traj['dones'][start_idx: end_idx].reshape(1, -1))
            timesteps.append(np.arange(end_idx-s[-1].shape[1], end_idx ).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            timesteps[-1][timesteps[-1] <= 0] = 0  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][start_idx:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([np.zeros((1, 1, 1)), rtg[-1]], axis=1)

            # TODO: padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, length - tlen, self.state_dim)), s[-1]], axis=1)
            # s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, length - tlen, self.action_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, length - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, length - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, length - tlen, 1)), rtg[-1]], axis=1)  # / scale
            timesteps[-1] = np.concatenate([np.zeros((1, length - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, length - tlen)), np.ones((1, tlen))], axis=1))
        #print(f"{time.time()}==={batch_idx[0]}", time.time()-start_time)
        return s, a, r, d, rtg, timesteps, mask

    def move_data_to_device(self, s, a, r, d, rtg, timesteps, mask):
        s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(a).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(r).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(d).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(mask).to(device=self.device)

        # s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=self.device)
        # a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=self.device)
        # r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=self.device)
        # d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=self.device)
        # rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=self.device)
        # timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=self.device)
        # mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)

        return s, a, r, d, rtg, timesteps, mask

    def sample_end_data(self, batch_size=256, length=30):
        batch_idx = np.random.choice(np.arange(len(self.data)), size=batch_size, p=self.p)
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for idx in batch_idx:
            traj = self.data[idx]
            start_idx = np.random.randint(0, len(traj['observations']) - 1)

            # get sequences from dataset
            s.append(traj['observations'][start_idx:start_idx + length].reshape(1, -1, self.state_dim))
            a.append(traj['actions'][start_idx:start_idx + length].reshape(1, -1, self.action_dim))
            r.append(traj['rewards'][start_idx:start_idx + length].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][start_idx:start_idx + length].reshape(1, -1))
            else:
                d.append(traj['dones'][start_idx:start_idx + length].reshape(1, -1))
            timesteps.append(np.arange(start_idx, start_idx + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            timesteps[-1][timesteps[-1] <= 0] = 0  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][start_idx:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # TODO: padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, length - tlen, self.state_dim)), s[-1]], axis=1)
            # s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, length - tlen, self.action_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, length - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, length - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, length - tlen, 1)), rtg[-1]], axis=1) #/ scale
            timesteps[-1] = np.concatenate([np.zeros((1, length - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, length - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)

        return s, a, r, d, rtg, timesteps, mask

class StateActionReturnDataset(Dataset):

    def __init__(self, global_state, block_size, actions, done_idxs, rewards, avas, v_values, rtgs, rets,
                 advs, timesteps):
        self.block_size = block_size
        self.global_state = global_state
        self.actions = actions
        self.done_idxs = done_idxs
        self.rewards = rewards
        self.avas = avas
        self.v_values = v_values
        self.rtgs = rtgs
        self.rets = rets
        self.advs = advs
        self.timesteps = timesteps

    def __len__(self):
        # return len(self.global_state) - self.block_size
        return len(self.global_state)

    def stats(self):
        print("max episode length: ", max(np.array(self.done_idxs[1:]) - np.array(self.done_idxs[:-1])))
        print("min episode length: ", min(np.array(self.done_idxs[1:]) - np.array(self.done_idxs[:-1])))
        print("max rtgs: ", max(self.rtgs))
        print("aver episode rtgs: ", np.mean([self.rtgs[i] for i in self.done_idxs[:-1]]))

    @property
    def max_rtgs(self):
        return max(self.rtgs)[0]

    def __getitem__(self, idx):
        context_length = self.block_size // 3
        done_idx = idx + context_length
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - context_length
        states = torch.tensor(np.array(self.global_state[idx:done_idx]), dtype=torch.float32)
        obss = torch.tensor(np.array(self.local_obs[idx:done_idx]), dtype=torch.float32)

        if done_idx in self.done_idxs:
            next_states = [np.zeros_like(self.global_state[idx]).tolist()] + self.global_state[idx+1:done_idx] + \
                          [np.zeros_like(self.global_state[idx]).tolist()]
            next_states.pop(0)
            next_rtgs = [np.zeros_like(self.rtgs[idx]).tolist()] + self.rtgs[idx+1:done_idx] + \
                        [np.zeros_like(self.rtgs[idx]).tolist()]
            next_rtgs.pop(0)
        else:
            next_states = self.global_state[idx+1:done_idx+1]
            next_rtgs = self.rtgs[idx+1:done_idx+1]
        next_states = torch.tensor(next_states, dtype=torch.float32)
        next_rtgs = torch.tensor(next_rtgs, dtype=torch.float32)

        if idx == 0 or idx in self.done_idxs:
            pre_actions = [[0]] + self.actions[idx:done_idx-1]
        else:
            pre_actions = self.actions[idx-1:done_idx-1]
        pre_actions = torch.tensor(pre_actions, dtype=torch.long)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long)

        rewards = torch.tensor(self.rewards[idx:done_idx], dtype=torch.float32)
        v_values = torch.tensor(self.v_values[idx:done_idx], dtype=torch.float32)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32)
        rets = torch.tensor(self.rets[idx:done_idx], dtype=torch.float32)
        advs = torch.tensor(self.advs[idx:done_idx], dtype=torch.float32)
        # timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64)
        timesteps = torch.tensor(self.timesteps[idx:done_idx], dtype=torch.int64)

        dones = torch.zeros_like(rewards)
        if done_idx in self.done_idxs:
            dones[-1][0] = 1
        
        return states, obss, actions, rewards, v_values, rtgs, rets, advs, timesteps, pre_actions, next_states, next_rtgs, dones

class SingleBuffer:
    def __init__(self, state_dim, action_dim,config):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = config['buffer_size']

    def insert(self, state, reward, action, done, value=None, logp=None):
        self.data = []
        self.episodes = []
        self.episode_dones = []


class OnlineBuffer:
    def __init__(self, block_size, global_obs_dim, action_dim, config):
        self.block_size = block_size
        self.buffer_size = 5000
        self.global_obs_dim = global_obs_dim
        self.action_dim = action_dim
        self.config = config
        self.data = []
        self.episodes = []
        self.episode_dones = []
        self.episodes_mapname = []
        self.gamma = config['online']['gamma']
        self.gae_lambda = config['online']['gae_lambda']
        self.client = config['client']
        self.offPolicyRate = config['online']['offpolicyRate']
        self.v_trace = config['online']['v_trace']

    @property
    def size(self):
        return len(self.data)

    def process_episode(self, episode):
        # episode: n_agent * timestep * [state, action, reward, done, available_action, v_value, logp]
        state = np.array(np.array(episode)[:,:,0].tolist())
        action = np.array(np.array(episode)[:,:,2].tolist())
        reward = np.array(np.array(episode)[:,:,3].tolist())
        done = np.array(np.array(episode)[:,:,4].tolist())
        available_action = np.array(np.array(episode)[:,:,5].tolist())
        v_value = np.array(np.array(episode)[:,:,6].tolist())
        logp = np.array(np.array(episode)[:,:,7].tolist())
        return state, action, reward, done, available_action, v_value, logp

    def insert(self, global_obs, action, reward, done, available_actions, v_value, logp, mapname=None):
        n_threads, n_agents = np.shape(reward)[0], np.shape(reward)[1]
        for n in range(n_threads):
            if len(self.episodes) < n + 1:
                self.episodes.append([])
                self.episode_dones.append(False)
                self.episodes_mapname.append(mapname)
            if not self.episode_dones[n]:
                for i in range(n_agents):
                    if len(self.episodes[n]) < i + 1:
                        self.episodes[n].append([])
                    step = [global_obs[n][i].tolist(),  action[n][i].tolist(),
                            reward[n][i].tolist(), done[n][i], available_actions[n][i].tolist(), v_value[n][i].tolist(), logp[n][i]]
                    self.episodes[n][i].append(step)
                if np.all(done[n]):
                    self.episode_dones[n] = True
                    if self.size > self.buffer_size:
                        raise NotImplementedError
                    if self.size == self.buffer_size:
                        del self.data[0]
                    temp = [np.array(self.episodes_mapname[n])]
                    temp.extend(self.process_episode(copy.deepcopy(self.episodes[n])))
                    self.data.append(temp)
        if np.all(self.episode_dones):
            self.episodes = []
            self.episode_dones = []
            self.episodes_mapname = []

    def loadOffPolicyData(self):
        print("\n download offpolicy data:")
        if self.size * self.offPolicyRate >0:
            off_data = list(self.client.sample('my_table', num_samples=self.size * self.offPolicyRate))
            for i in off_data:
                self.data.append(i[0].data)


    def uploadOnlineData(self):
        print("\n upload online data:")
        for i in tqdm(range(self.size)):
            self.client.insert(self.data[i], priorities={'my_table': 1.0})

    def reset(self, num_keep=0, buffer_size=5000):
        self.buffer_size = buffer_size
        if num_keep == 0:
            self.data = []
        elif self.size >= num_keep:
            keep_idx = np.random.randint(0, self.size, num_keep)
            self.data = [self.data[idx] for idx in keep_idx]

    def sample_raw(self):
        return self.data

    def sample_ppo(self):
        # adding elements with list will be faster
        global_states = []
        local_obss = []
        actions = []
        rewards = []
        v_values = []
        rtgs = []
        rets = []
        done_idxs = []
        time_steps = []
        advs = []
        logps = []

        for episode_idx in tqdm(range(self.size)):
            episode = self.get_episode(episode_idx)
            # episode = self.get_episode(episode_idx, min_return)
            if episode is None:
                continue
            for agent_trajectory in episode:
                time_step = 0
                for step in agent_trajectory:
                    g, a, r, d, v, logp, rtg, ret, adv = step
                    global_states.append(g)
                    actions.append(a)
                    rewards.append(r)
                    v_values.append(v)
                    rtgs.append(rtg)
                    rets.append(ret)
                    advs.append(adv)
                    logps.append(logp)
                    time_steps.append([time_step])
                    time_step += 1
                # done_idx - 1 equals the last step's position
                done_idxs.append(len(global_states))

        # or we can separate it as well
        # states = np.concatenate((global_states, local_obss), axis=1)
        dataset = StateActionReturnDataset(global_states, self.block_size, actions, done_idxs, rewards,
                                            v_values, rtgs, rets, advs, time_steps)
        return dataset


    def sample(self):
        if self.v_trace:
            return  self.sample_raw()
        else:
            return  self.sample_ppo()

    # from [g, o, a, r, d, ava, logp]/[g, o, a, r, d, ava, v, logp] to [g, o, a, r, d, ava, v, logp, rtg, ret, adv]
    def get_episode(self, index):
        episode = copy.deepcopy(self.data[index])

        # cal rtg and ret
        for agent_trajectory in episode:
            rtg = 0.
            ret = 0.
            adv = 0.
            for i in reversed(range(len(agent_trajectory))):
                if len(agent_trajectory[i]) == 6:  # offline, give a fake v_value, unused
                    agent_trajectory[i].append([0.])
                    raise NotImplementedError #TODO: since add the logp, this data term order have changed
                elif len(agent_trajectory[i]) == 8:
                    pass  # online nothing to do
                else:
                    raise NotImplementedError

                reward = agent_trajectory[i][3][0]
                rtg += reward
                agent_trajectory[i].append([rtg])

                # todo: check ret and adv calculation
                if i == len(agent_trajectory) - 1:
                    next_v = 0.
                else:
                    next_v = agent_trajectory[i + 1][6][0]
                v = agent_trajectory[i][6][0]
                # adv with gae
                delta = reward + self.gamma * next_v - v
                adv = delta + self.gamma * self.gae_lambda * adv

                # adv without gae
                # adv = reward + self.gamma * next_v - v

                # ret = adv + v
                ret = reward + self.gamma * ret
                # ret = reward + self.gamma * next_v
                # print("reward: %s, v: %s, next_v: %s, adv: %s, ret: %s " % (reward, v, next_v, adv, ret))

                agent_trajectory[i].append([ret])
                agent_trajectory[i].append([adv])

        # prune dead steps
        for i in range(len(episode)):
            end_idx = 0
            for step in episode[i]:
                if step[4]:
                    break
                else:
                    end_idx += 1
            episode[i] = episode[i][0:end_idx + 1]
        return episode






class ReplayBuffer:

    def __init__(self, block_size, global_obs_dim, local_obs_dim, action_dim):
        self.block_size = block_size
        self.buffer_size = 5000
        self.global_obs_dim = global_obs_dim
        self.local_obs_dim = local_obs_dim
        self.action_dim = action_dim
        self.data = []
        self.episodes = []
        self.episode_dones = []
        self.gamma = 0.99
        self.gae_lambda = 0.95

    @property
    def size(self):
        return len(self.data)

    def insert(self, global_obs, local_obs, action, reward, done, available_actions, v_value):
        n_threads, n_agents = np.shape(reward)[0], np.shape(reward)[1]
        for n in range(n_threads):
            if len(self.episodes) < n + 1:
                self.episodes.append([])
                self.episode_dones.append(False)
            if not self.episode_dones[n]:
                for i in range(n_agents):
                    if len(self.episodes[n]) < i + 1:
                        self.episodes[n].append([])
                    step = [global_obs[n][i].tolist(), local_obs[n][i].tolist(), action[n][i].tolist(),
                            reward[n][i].tolist(), done[n][i], available_actions[n][i].tolist(), v_value[n][i].tolist()]
                    self.episodes[n][i].append(step)
                if np.all(done[n]):
                    self.episode_dones[n] = True
                    if self.size > self.buffer_size:
                        raise NotImplementedError
                    if self.size == self.buffer_size:
                        del self.data[0]
                    self.data.append(copy.deepcopy(self.episodes[n]))
        if np.all(self.episode_dones):
            self.episodes = []
            self.episode_dones = []

    def reset(self, num_keep=0, buffer_size=5000):
        self.buffer_size = buffer_size
        if num_keep == 0:
            self.data = []
        elif self.size >= num_keep:
            keep_idx = np.random.randint(0, self.size, num_keep)
            self.data = [self.data[idx] for idx in keep_idx]

    # offline data size could be large than buffer size
    def load_offline_data(self, data_dir, offline_episode_num):
        episode_idx = 0
        for j in range(len(data_dir)):
            path_files = glob.glob(pathname=data_dir[j] + "*")
            # for file in sorted(path_files):
            for i in range(offline_episode_num[j]):
                episode = torch.load(path_files[i])

                # padding obs
                for agent_trajectory in episode:
                    for step in agent_trajectory:
                        step[0] = padding_obs(step[0], self.global_obs_dim)
                        step[1] = padding_obs(step[1], self.local_obs_dim)
                        step[5] = padding_obs(step[5], self.action_dim)
                print(f"Episode index {episode_idx}")
                episode_idx += 1
                self.data.append(episode)

    def sample(self):
        # adding elements with list will be faster
        global_states = []
        local_obss = []
        actions = []
        rewards = []
        avas = []
        v_values = []
        rtgs = []
        rets = []
        done_idxs = []
        time_steps = []
        advs = []

        for episode_idx in tqdm(range(self.size)):
            episode = self.get_episode(episode_idx)
            # episode = self.get_episode(episode_idx, min_return)
            if episode is None:
                continue
            for agent_trajectory in episode:
                time_step = 0
                for step in agent_trajectory:
                    g, o, a, r, d, ava, v, rtg, ret, adv = step
                    global_states.append(g)
                    local_obss.append(o)
                    actions.append(a)
                    rewards.append(r)
                    avas.append(ava)
                    v_values.append(v)
                    rtgs.append(rtg)
                    rets.append(ret)
                    advs.append(adv)
                    time_steps.append([time_step])
                    time_step += 1
                # done_idx - 1 equals the last step's position
                done_idxs.append(len(global_states))

        # or we can separate it as well
        # states = np.concatenate((global_states, local_obss), axis=1)
        dataset = StateActionReturnDataset(global_states, local_obss, self.block_size, actions, done_idxs, rewards,
                                           avas, v_values, rtgs, rets, advs, time_steps)
        return dataset

    # from [g, o, a, r, d, ava]/[g, o, a, r, d, ava, v] to [g, o, a, r, d, ava, v, rtg, ret, adv]
    def get_episode(self, index):
        episode = copy.deepcopy(self.data[index])

        # cal rtg and ret
        for agent_trajectory in episode:
            rtg = 0.
            ret = 0.
            adv = 0.
            for i in reversed(range(len(agent_trajectory))):
                if len(agent_trajectory[i]) == 6:  # offline, give a fake v_value, unused
                    agent_trajectory[i].append([0.])
                elif len(agent_trajectory[i]) == 7:
                    pass  # online nothing to do
                else:
                    raise NotImplementedError

                reward = agent_trajectory[i][3][0]
                rtg += reward
                agent_trajectory[i].append([rtg])

                # todo: check ret and adv calculation
                if i == len(agent_trajectory) - 1:
                    next_v = 0.
                else:
                    next_v = agent_trajectory[i + 1][6][0]
                v = agent_trajectory[i][6][0]
                # adv with gae
                delta = reward + self.gamma * next_v - v
                adv = delta + self.gamma * self.gae_lambda * adv

                # adv without gae
                # adv = reward + self.gamma * next_v - v

                # ret = adv + v
                ret = reward + self.gamma * ret
                # ret = reward + self.gamma * next_v
                # print("reward: %s, v: %s, next_v: %s, adv: %s, ret: %s " % (reward, v, next_v, adv, ret))

                agent_trajectory[i].append([ret])
                agent_trajectory[i].append([adv])

        # prune dead steps
        for i in range(len(episode)):
            end_idx = 0
            for step in episode[i]:
                if step[4]:
                    break
                else:
                    end_idx += 1
            episode[i] = episode[i][0:end_idx + 1]
        return episode


class Replay_buffer(object):
    def __init__(self, max_size, trajectory_property):
        self.storage = []
        self.max_size = max_size

        self.property_list = ['states', 'states_next', 'rewards', 'dones', 'hidden', 'next_hidden', 'hidden_q', 'next_hidden_q', 'hidden_q_target', 'next_hidden_q_target']
        # self.property_list = ['states', 'states_next', 'rewards', 'dones']
        self.property_additional = trajectory_property
        self.properties_all = self.property_list + self.property_additional
        self.item_buffers = dict()
        self.step_index_by_env = 0

        self.buffer_dict = dict()
        self.buffer_dict_clear()
        self.ptr = 0

    def buffer_dict_clear(self):
        for item in self.properties_all:
            self.buffer_dict[item] = list()

    def init_item_buffers(self):
        for p in self.properties_all:
            self.item_buffers[p] = ItemBuffer(self.max_size, p)

    def insert(self, item_name:str, agent_id:int, data:np.ndarray, step=None):
        if item_name == 'dones':
            agent_id = 0
        self.item_buffers[item_name].insert(agent_id, step, data)

    def sample(self, batch_size):
        self.buffer_dict_clear()
        data_length = len(self.item_buffers["action"].data)
        ind = np.random.randint(0, data_length, size=batch_size)
        for name, item_buffer in self.item_buffers.items():
            for i in ind:
                self.buffer_dict[name].append(np.array(item_buffer.data[i], copy=False))
        return self.buffer_dict

    def sample2(self, batch_size):
        self.buffer_dict_clear()
        data_length = len(self.item_buffers["action"].data)
        for name, item_buffer in self.item_buffers.items():
            self.buffer_dict[name].append(np.array(item_buffer.data[-batch_size:], copy=False))
        return self.buffer_dict

    def get_trajectory(self):
        self.buffer_dict_clear()
        data_length = len(self.item_buffers["action"].data)
        for name, item_buffer in self.item_buffers.items():
            for i in range(data_length):
                self.buffer_dict[name].append(np.array(item_buffer.data[i], copy=False))
        return self.buffer_dict

    def get_step_data(self):
        self.buffer_dict_clear()
        for name, item_buffer in self.item_buffers.items():
            self.buffer_dict[name] = item_buffer.data[0]
        return self.buffer_dict

    def item_buffer_clear(self):
        for p in self.properties_all:
            self.item_buffers[p].clear()


class ItemBuffer(object):
    def __init__(self, max_size, name):
        self.name = name
        self.max_size = max_size
        self.A = 1
        self.data = list()
        self.ptr = 0

    def insert(self, agent_id:int, step:int, data:np.ndarray):
        if len(self.data) == self.max_size:
            self.data.pop(0)
        self.data.append(data)

    def clear(self):
        del self.data[:]

