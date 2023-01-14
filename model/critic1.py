import torch
import torch.nn as nn
import torch.nn.functional as F

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
    #  np.random.seed(seed)
    #  random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(7)

class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layer=0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_layer = num_hidden_layer

        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)
        if self.num_hidden_layer > 0:
            hid_net = []
            for _ in range(self.num_hidden_layer):
                hid_net.append(nn.Linear(hidden_size, hidden_size))
                hid_net.append(nn.ReLU())
            self.linear_hid = nn.Sequential(*hid_net)

    def forward(self, x):
        x = F.relu(self.linear_in(x))
        if self.num_hidden_layer > 0:
            x = self.linear_hid(x)
        x = self.linear_out(x)
        return x


class RCritic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layer=0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_layer = num_hidden_layer

        self.linear_in = nn.Linear(input_size, hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        for name, param in self.rnn.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
        self.linear_out = nn.Linear(hidden_size, output_size)
        if self.num_hidden_layer > 0:
            hid_net = []
            for _ in range(self.num_hidden_layer):
                hid_net.append(nn.Linear(hidden_size, hidden_size))
                hid_net.append(nn.ReLU())
            self.linear_hid = nn.Sequential(*hid_net)
        self.h = None

    def forward(self, x):
        self.hidden_q = self.h
        x = F.relu(self.linear_in(x))
        # print("Critic: x_shape=", x.shape)
        x, self.h = self.rnn(x, self.h)
        if self.num_hidden_layer > 0:
            x = self.linear_hid(x)
        x = self.linear_out(x)
        self.next_hidden_q = self.h
        self.hidden_q = self.hidden_q if self.hidden_q != None else torch.zeros_like(self.next_hidden_q)
        return x
    
    def set_h(self, h=None):
        self.h = h