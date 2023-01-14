"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    # embd_pdrop = 0.1
    # resid_pdrop = 0.1
    # attn_pdrop = 0.1
    embd_pdrop = 0.
    resid_pdrop = 0.
    attn_pdrop = 0.

    def __init__(self, state_size, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.state_size = state_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd) #TODO: [Lin] attention have bias term ?
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                             .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        B2, T2, C2 = layer_past.size() if layer_past != None else x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim

        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        if layer_past == None:
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        else:
            # layer_past as memory
            k = self.key(layer_past).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T2, hs)
            v = self.value(layer_past).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T2, hs)

        # causal self-attention; mem-attend: (B, nh, T, hs) x (B, nh, hs, T2) -> (B, nh, T, T2)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past!=None:
            # mem attention,  do not mask the memory
            pass
        else:
            # encoder and decoder both masked
            att = att.masked_fill(self.mask[:, :, :T, :T2] == 0, float('-inf')) #TODO: [Lin] cheak the mask for first decode attention
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y



class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 2 * config.n_embd),
            GELU(),
            nn.Linear(2 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class DecodeBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)

        self.mask_attn = CausalSelfAttention(config)
        self.memory_attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 2 * config.n_embd),
            GELU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(2 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x, mem = x
        x = x + self.mask_attn(self.ln1(x))
        x = x + self.memory_attn(self.ln2(x), mem)
        x = x + self.mlp(self.ln3(x))
        return [x, mem]



class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, model_type='actor'):
        super().__init__()

        self.config = config
        self.para_config = config.config

        self.model_type = config.model_type
        self.state_size = config.state_size

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        if model_type == 'actor':
            self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        elif model_type == 'critic':
            if self.para_config['online']['popart']:
                self.head = PopArtLayer(config.n_embd, 1, beta=self.para_config['online']['popart_beta'], task_ls=self.para_config['online']['map_lists'])
            else:
                self.head = nn.Linear(config.n_embd, 1, bias=False)
        else:
            raise NotImplementedError

        self.block_size = config.block_size
        self.apply(self._init_weights)

        self.parameter_number = sum(p.numel() for p in self.parameters())
        logger.info("number of parameters: %e", self.parameter_number)

        # self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
        #                                    nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
        #                                    nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
        #                                    nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh())

        # self.state_encoder = nn.Sequential(nn.Linear(self.state_size, 128), nn.ReLU(),
        #                                    nn.Linear(128, 64), nn.ReLU(),
        #                                    nn.Linear(64, config.n_embd), nn.Tanh())

        self.state_encoder = nn.Sequential(nn.Linear(self.state_size, config.n_embd), nn.Tanh())

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.mask_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config, lr):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.startswith('head') and pn.find('ls') > -1:
                    if pn.find('weight') > -1:
                        decay.add(pn)
                    elif pn.find('bias') > -1:
                        no_decay.add(pn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, pre_actions, rtgs=None, timesteps=None, map_name=None):
        # states: (batch, context_length, 4*84*84)
        # actions: (batch, context_length, 1)
        # targets: (batch, context_length, 1)
        # rtgs: (batch, context_length, 1)
        # timesteps: (batch, context_length, 1)

        state_embeddings = self.state_encoder(states.reshape(-1, self.state_size).type(torch.float32).contiguous())
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1],
                                                    self.config.n_embd)  # (batch, block_size, n_embd)

        if self.model_type == 'rtgs_state_action':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            action_embeddings = self.action_embeddings(
                pre_actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 3, self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings
            num_elements = 3
        elif self.model_type == 'state_action':
            action_embeddings = self.action_embeddings(
                pre_actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 2, self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings
            num_elements = 2
        elif self.model_type == 'state_only':
            token_embeddings = state_embeddings
            num_elements = 1
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0)
        global_pos_emb = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1))
        global_pos_emb = torch.repeat_interleave(global_pos_emb, num_elements, dim=1)
        context_pos_emb = self.pos_emb[:, :token_embeddings.shape[1], :]
        position_embeddings = global_pos_emb + context_pos_emb

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x) if map_name == None else self.head(x, map_name) # for popart head, should give the map_name.

        if self.model_type == 'rtgs_state_action':
            # logits = logits[:, 1::3, :]  # only keep predictions from state_embeddings
            logits = logits[:, 2::3, :]  # consider all tokens
        elif self.model_type == 'state_action':
            # logits = logits[:, ::2, :]  # only keep predictions from state_embeddings
            logits = logits[:, 1::2, :]  # consider all tokens
        elif self.model_type == 'state_only':
            logits = logits
        else:
            raise NotImplementedError()

        return logits


class PopArtLayer(torch.nn.Module):

    def __init__(self, input_features, output_features, beta=4e-4, task_ls=[0]):
        self.beta = beta

        super(PopArtLayer, self).__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.task_ls = task_ls
        self.task_idx_dict = {}
        for i in range(len(task_ls)):
            self.task_idx_dict[task_ls[i]] = i

        self.weight_ls = nn.ParameterList([torch.nn.Parameter(torch.Tensor(output_features, input_features)) for i in range(len(task_ls))])
        self.bias_ls = nn.ParameterList( [torch.nn.Parameter(torch.Tensor(output_features)) for i in range(len(task_ls))])

        self.register_buffer('mu', torch.zeros(len(task_ls), output_features, requires_grad=False))
        self.register_buffer('sigma', torch.ones(len(task_ls), output_features, requires_grad=False))

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.task_ls)):
            weight, bias = self.weight_ls[i], self.bias_ls[i]
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            if bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(bias, -bound, bound)

    def normalize(self, data, task):
        task_idx = self.task_idx_dict[task]
        data = (data - self.mu[task_idx,:]) / self.sigma[task_idx,:]
        return data

    def forward(self, inputs, map_name=None):
        # input: batch_size * context_length * dim
        if map_name != None:
            task_idx = self.task_idx_dict[map_name]
            weight = self.weight_ls[task_idx]
            bias = self.bias_ls[task_idx]
            normalized_output = torch.matmul(inputs, weight.t())
            normalized_output += bias.unsqueeze(0).expand_as(normalized_output)

            with torch.no_grad():
                output = normalized_output * self.sigma[task_idx, :] + self.mu[task_idx, :]

            return [output, normalized_output]
        else:
            # just inference record which is useless
            #TODO: [Lin] check if the value have been used
            task_idx = 0
            weight = self.weight_ls[task_idx]
            bias = self.bias_ls[task_idx]
            normalized_output = torch.matmul(inputs, weight.t())
            normalized_output += bias.unsqueeze(0).expand_as(normalized_output)
            return normalized_output

    def update_parameters(self, vs, map_name=None):
        # vs: should have filtered by done
        task_idx = self.task_idx_dict[map_name]

        oldmu = self.mu[task_idx, :]
        oldsigma = self.sigma[task_idx, :]

        n = vs.shape[0]
        mu = vs.sum()/ vs.shape[0]
        nu = torch.sum(vs**2) / n
        sigma = torch.sqrt(nu - mu**2)
        sigma = torch.clamp(sigma, min=1e-4, max=1e+6)

        mu = self.mu[task_idx, :] if torch.isnan(mu) else mu#mu[torch.isnan(mu)] = self.mu[task_idx, torch.isnan(mu)]
        sigma = self.sigma[task_idx, :] if torch.isnan(sigma) else sigma#sigma[torch.isnan(sigma)] = self.sigma[task_idx, torch.isnan(sigma)]

        self.mu[task_idx, :] = (1 - self.beta) * self.mu[task_idx, :] + self.beta * mu
        self.sigma[task_idx, :] = (1 - self.beta) * self.sigma[task_idx, :] + self.beta * sigma

        self.weight_ls[task_idx].data = (self.weight_ls[task_idx].t() * oldsigma / self.sigma[task_idx,:]).t()
        self.bias_ls[task_idx].data = (oldsigma * self.bias_ls[task_idx] + oldmu - self.mu[task_idx,:]) / self.sigma[task_idx,:]


class MADT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, model_type='actor'):
        super().__init__()

        self.config = config
        self.para_config = config.config

        self.model_type = config.model_type
        self.state_size = config.state_size
        self.obs_size = config.local_obs_dim

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # decoder - value head
        self.ln_f = nn.LayerNorm(config.n_embd)
        if self.para_config['online']['popart']:
            self.head = PopArtLayer(config.n_embd, 1, beta=self.para_config['online']['popart_beta'], task_ls=self.para_config['online']['map_lists'])
        else:
            self.head = nn.Linear(config.n_embd, 1, bias=False)


        # decoder - action head
        self.decode_blocks = nn.Sequential(*[DecodeBlock(config) for _ in range(config.n_layer)])
        self.actor_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        self.parameter_number = sum(p.numel() for p in self.parameters())
        logger.info("number of parameters: %e", self.parameter_number)

        # self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
        #                                    nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
        #                                    nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
        #                                    nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh())

        # self.state_encoder = nn.Sequential(nn.Linear(self.state_size, 128), nn.ReLU(),
        #                                    nn.Linear(128, 64), nn.ReLU(),
        #                                    nn.Linear(64, config.n_embd), nn.Tanh())

        self.state_encoder = nn.Sequential(nn.Linear(self.state_size, config.n_embd), nn.Tanh())
        self.action_encoder = nn.Sequential(nn.Linear(config.vocab_size, config.n_embd), nn.Tanh())

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.mask_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config, lr):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.startswith('head') and pn.find('ls') > -1:
                    if pn.find('weight') > -1:
                        decay.add(pn)
                    elif pn.find('bias') > -1:
                        no_decay.add(pn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, observation, pre_actions, rtgs=None, timesteps=None, map_name=None, history_action=None, available_actions=None):
        # states: (batch, agent_number, 4*84*84)
        # actions: (batch, agent_number, 1)
        # targets: (batch, agent_number, 1)
        # rtgs: (batch, agent_number, 1)
        # timesteps: (batch, agent_number, 1)
        
        
        state_embeddings = self.state_encoder(
            states.reshape(-1, self.state_size).type(torch.float32).contiguous())
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1],
                                                    self.config.n_embd)  # (batch, block_size, n_embd)

        if self.model_type == 'rtgs_state_action':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            action_embeddings = self.action_embeddings(
                pre_actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 3, self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings
            num_elements = 3
        elif self.model_type == 'state_action':
            action_embeddings = self.action_embeddings(
                pre_actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 2, self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings
            num_elements = 2
        elif self.model_type == 'state_only':
            token_embeddings = state_embeddings
            num_elements = 1
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0)
        global_pos_emb = all_global_pos_emb[:,:states.shape[1],:]
        global_pos_emb = torch.repeat_interleave(global_pos_emb, num_elements, dim=1)
        # context_pos_emb = self.pos_emb[:, :token_embeddings.shape[1], :]
        position_embeddings = global_pos_emb # + context_pos_emb

        # encode there is no need to add position embedding, move to decode
        x = self.drop(token_embeddings) + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)


        value = self.head(x) if map_name == None else self.head(x, map_name)

        if self.model_type == 'rtgs_state_action':
            # logits = logits[:, 1::3, :]  # only keep predictions from state_embeddings
            value = value[:, 2::3, :]  # consider all tokens
        elif self.model_type == 'state_action':
            # logits = logits[:, ::2, :]  # only keep predictions from state_embeddings
            value = value[:, 1::2, :]  # consider all tokens
        elif self.model_type == 'state_only':
            value = value
        else:
            raise NotImplementedError()

        # decode for actions
        start_op = torch.zeros(states.shape[0], 1, self.config.vocab_size).to(self.device)
        if history_action != None:
            # for training, using history action to inference logits
            input_decoder_token = torch.nn.functional.one_hot(history_action[:,:-1,-1], num_classes=self.config.vocab_size).float() # for the last action not need to input
            input_decoder_token = torch.cat((start_op, input_decoder_token), dim=1)
            input_decoder = self.action_encoder(input_decoder_token)
            #input_decoder += all_global_pos_emb[:,:input_decoder.shape[1],:]
            output_op, _ = self.decode_blocks([input_decoder, x])
            action = history_action
        else:
            # for sample, should sample action recursive during training
            input_decoder_token = start_op
            for i in range(states.shape[1]):
                input_decoder = self.action_encoder(input_decoder_token)
                #input_decoder += all_global_pos_emb[:, :input_decoder.shape[1], :]
                output_op, _ = self.decode_blocks([input_decoder, x])
                logits = self.actor_head(output_op)

                if available_actions is not None:
                    logits[available_actions[:,:logits.shape[1],:] == 0] = -1e10
                probs = F.softmax(logits, dim=-1)

                if True: #TODO: modify sample when training random, when eval topk
                    a = torch.multinomial(probs[:, -1, :], num_samples=1)
                else:
                    _, a = torch.topk(probs[:, -1, :], k=1, dim=-1)

                # set the first a to action, and then concat the new action
                a = torch.nn.functional.one_hot(a, num_classes=self.config.vocab_size)
                input_decoder_token = torch.cat((input_decoder_token, a[:,-1:,:]), 1)
            # drop the start op
            action_one_hot = input_decoder_token[:,1:,:]
            action = torch.argmax(action_one_hot, dim=2)

        logits = self.actor_head(output_op)
        return logits, value, action