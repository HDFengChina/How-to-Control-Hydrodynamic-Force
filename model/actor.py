import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal
import wandb


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = torch.tensor(1e-6, device='cuda')

class GaussianActor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, tanh=False, action_high = 10, action_low = -10):
        super(GaussianActor, self).__init__()

        self.linear_in = nn.Linear(state_dim, hidden_dim)
        self.linear_hid = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.logstd_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

        self.tanh = tanh
        if tanh:  # normalise the action
            self.action_scale = torch.FloatTensor([(action_high - action_low) / 2.]).to('cuda:0')
            self.action_bias = torch.FloatTensor([(action_high + action_low) / 2.]).to('cuda:0')

    def forward(self, state):
        x = F.relu(self.linear_in(state))
        x = F.relu(self.linear_hid(x))
        mean = self.mean_linear(x)
        #mean = torch.clamp(mean, min=-1.2, max=1.2) #TODO: tune the paramerter, and load from config.yaml
        log_std = self.logstd_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, logstd = self.forward(state)
        std = logstd.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        #x_t = torch.clamp(x_t, min=-1.2, max=1.2) #TODO: tune the paramerter, and load from config.yaml
        if self.tanh:
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            action = x_t
            log_prob = normal.log_prob(x_t)
            log_prob = log_prob.sum(1, keepdim = True)
            mean = mean

        return action, log_prob, mean


class RActor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, tanh=True, action_high = 1, action_low = -1):
        super(RActor, self).__init__()

        self.linear_in = nn.Linear(state_dim, hidden_dim)
        self.linear_hid = nn.Linear(hidden_dim, hidden_dim)
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True) #batch * time_len * dim
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.logstd_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)
        self.h = None
        self.device = 'cuda'

        self.tanh = tanh
        if tanh:  # normalise the action
            self.action_scale = torch.FloatTensor([(action_high - action_low) / 2.]).to(self.device)
            self.action_bias = torch.FloatTensor([(action_high + action_low) / 2.]).to(self.device)

    def forward(self, state):
        x = F.relu(self.linear_in(state))
        # x = F.relu(self.linear_hid(x))
        # print("x=", x.unsqueeze(0), "self.h=", self.h)
        # print("x_shape=", x.shape)
        x, self.h = self.rnn(x, self.h)
        mean = self.mean_linear(x)
        log_std = self.logstd_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        hidden = self.h
        mean, logstd = self.forward(state)
        next_hidden = self.h
        hidden = hidden if hidden != None else torch.zeros_like(next_hidden)
        std = logstd.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        if self.tanh:
            #y_t = torch.tanh(x_t).to(self.device)
            #action = y_t * self.action_scale + self.action_bias
            #log_prob = normal.log_prob(x_t).to(self.device)
            #log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
            # TODO
            #log_prob = log_prob.sum(2, keepdim=True)
            # log_prob = log_prob.sum(1, keepdim=True)

            log_prob = normal.log_prob(x_t).to(self.device)
            action = torch.tanh(x_t)
            log_prob -= torch.log(1-torch.tanh(action).pow(2)+epsilon)
            log_prob = log_prob.sum(2, keepdim=True)
            action = action * self.action_scale + self.action_bias

            mean = torch.tanh(mean).to(self.device) * self.action_scale + self.action_bias
        else:
            action = x_t
            log_prob = normal.log_prob(x_t)
            log_prob = log_prob.sum(2, keepdim = True)
            mean = mean

        return action, log_prob, mean, hidden.detach().squeeze(0).cpu().numpy(), next_hidden.detach().squeeze(0).cpu().numpy()

    def set_h(self, h=None):
        self.h = h


