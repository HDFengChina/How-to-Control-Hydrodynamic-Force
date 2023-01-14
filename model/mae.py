import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.distributions import Independent


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

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MaskAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., max_block =256):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.register_buffer("mask", rearrange(torch.tril(torch.ones(max_block+1, max_block+1)), 'a b -> 1 1 a b'))

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # batch timestep context_dim
        B, T, C = x.size()
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # batch * time * head * dim
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # TODO: Check the mask implementation
        # TODO:block should > 3* n_agent + 1
        dots = dots.masked_fill(self.mask[:,:, :T, :T] == 0, float('-inf'))
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class maskedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MaskAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class TransformerAgent(nn.Module):
    def __init__(
        self,
        config,
        decoder_dim_head = 64
    ):
        super().__init__()
        # construct encoder
        self.config = config
        self.n_agent = config.context_len # self.n_agent means the max input block
        self.action_dim = config.action_space if config.mode == 'actor' else 1# since mean and std are concatenated
        self.obs_dim = config.obs_space
        self.device = config.device
        self.action_max = config.action_max
        self.sigma_min = -5
        self.sigma_max = 0.5

        self.time_step = self.n_agent
        self.device = config.device
        self.mode = config.mode

        self.embed_dim = config.n_embed
        self.encoder_head = config.n_head
        self.encoder_dim = config.n_embed
        self.encoder_layer = config.n_layer
        #self.mode = config.mode #'oa' 'oar' 'oaro'
        self.encoder = Transformer(self.encoder_dim,
                                   self.encoder_layer,
                                   self.encoder_head,
                                   decoder_dim_head,
                                   self.encoder_dim * 4) # TODO:dropout, dim_head

        # token [TODO: all the same or different?]
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        #self.sigma_param = nn.Parameter(torch.ones(self.action_dim)*-2)#.to(self.device)
        self.register_parameter(name='sigma_param', param=nn.Parameter(torch.ones(self.action_dim)*-2))

        # position embeding
        self.obs_pos_embedding = nn.Parameter(torch.randn(1, self.time_step, self.embed_dim))
        self.timestep_embeding = nn.Embedding(self.time_step, self.embed_dim)

        # embed [TODO: is ReLu necessary?]
        self.to_obs_embed = nn.Sequential(
            nn.Linear(self.obs_dim, self.embed_dim)
        )

        self.embed_to_action = nn.Sequential(
            nn.Linear(self.embed_dim, self.action_dim)
        )

        # learn parameters
        self.optimizer = self.configure_optimizers()

        self.parameter_number = sum(p.numel() for p in self.parameters())
        print("number of parameters: %e", self.parameter_number)


    def forward(self, obs=None, train=False):
        """
        :param obs: [batch * n_agent/n_timestep * dim]
        :param action:
        :param reward:
        :param obs_next:
        :return: o,a,r after reconstruction
        """
        device = obs.device
        batch_num, n_timestep, _ = obs.shape

        # cal position embedding
        obs_pos_embedding = repeat(self.obs_pos_embedding, 'b n d -> (b b_repeat) n d', b_repeat=batch_num)

        # if mask, should use the specific token to replace the origin input
        # o-observation
        obs_token = self.to_obs_embed(obs)

        # TODO: check weather agent_postion_embeding is neccessary
        obs_token += obs_pos_embedding
        cls_token = repeat(self.cls_token, 'b t d -> (b b_repeat) t d', b_repeat=batch_num)

        # if input the current info or the info should as a results, should feed into the network.
        # TODO:[0809] cheak classfication
        #tokens_ls = [cls_token, obs_token]
        tokens_ls = [obs_token]
        tokens = torch.cat(tokens_ls, dim=1)

        # get the patches to be masked for the final reconstruction loss
        # attend with vision transformer [TODO: in CV, mlp head is used to merge infomation]
        encoded_tokens = self.encoder(tokens)

        # get the first token
        first_token = encoded_tokens[:, -1]
        action = self.embed_to_action(first_token)

        return action

    def reset_optimizer(self):
        self.optimizer = self.configure_optimizers()
        return self.optimizer

    def configure_optimizers(self):
        config = self.config
        #TODO: update optimaizer setting, identify the actor and critic
        # learning rate schedular
        if self.mode == 'actor':
            optimizer = torch.optim.AdamW(self.parameters(), lr=config.a_lr)#, betas=train_config.betas)
        elif self.mode == 'critic':
            optimizer = torch.optim.AdamW(self.parameters(), lr=config.c_lr)
        return optimizer

    def getValue(self, obs):
        re_flag = False
        if len(obs.shape) == 4:
            re_flag = True
            b = obs.shape[0]
            obs = rearrange(obs, "b t c d-> (b t) c d")

        # input dim [batch * time * dim]
        obs = obs[:, -self.n_agent:, :].to(device=self.device)

        value = self.forward(obs)
        if re_flag == True:
            value = rearrange(value, "(b t) 1 -> b t 1", b=b)
        return value

    def getAction(self, obs, train=False):
        # input dim [batch * time * dim]
        obs = obs[:, -self.n_agent:, :].to(device=self.device)

        a = self.forward(obs)
        # a = rearrange(a, 'b (s d)-> (b s) d', s=2)
        # mu, sigma = (a[:, 0], a[:, 1]
        mu, sigma = rearrange(a, 'b d-> (b d)'), repeat(self.sigma_param, 'd -> (b d)', b=a.shape[0])
        mu = self.action_max * torch.tanh(mu)
        sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max).exp()
        a_dis = torch.distributions.Normal(mu, sigma)
        a_ = a_dis.sample().detach().cpu().numpy()
        if train:
            a_log = a_dis.log_prob(a_).detach().cpu().numpy()
        else:
            a_log = None
        return a_, a_log

    def getActionLogProb(self, obs, action, train=False):
        re_flag = False
        if len(obs.shape) == 4:
            re_flag = True
            b = obs.shape[0]
            obs = rearrange(obs, "b t c d-> (b t) c d")
            action = rearrange(action, "b t d-> (b t) d")

        # input dim [batch * time * dim]
        obs = obs[:, -self.n_agent:, :].to(device=self.device)

        a = self.forward(obs)
        # a = rearrange(a, 'b (s d)-> (b s) d', s=2)
        # mu, sigma = (a[:, 0], a[:, 1]
        # [trick] tanh action
        mu, sigma = a, repeat(self.sigma_param, 'd -> b d', b=a.shape[0]).exp()
        mu = self.action_max * torch.tanh(mu)
        #sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max).exp()
        if train:
            a_dis = Independent(torch.distributions.Normal(mu, sigma), 1)
        else:
            a_dis = Independent(torch.distributions.Normal(mu.detach(), sigma.detach()), 1)
        a_log = a_dis.log_prob(action)

        if re_flag == True:
            a_log = rearrange(a_log, "(b t) -> b t", b=b)
        return a_log

    def getVecAction(self, obs, train=True):
        re_flag = False
        if len(obs.shape) == 4:
            re_flag = True
            b = obs.shape[0]
            obs = rearrange(obs, "b t c d-> (b t) c d")
        # input dim [batch * time * dim]
        obs = obs[:, -self.n_agent:, :].to(device=self.device)

        a = self.forward(obs)
        #a = rearrange(a, 'b (s d)-> (b s) d', s=2)
        #mu, sigma = (a[:, 0], a[:, 1])
        mu, sigma = a, repeat(self.sigma_param, 'd -> b d', b=a.shape[0]).exp()
        mu = self.action_max * torch.tanh(mu)
        #sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max).exp()
        a_dis = Independent(torch.distributions.Normal(mu, sigma), 1)
        a_ = a_dis.sample()
        self.entropy = a_dis.entropy().mean().item()
        if train:
            a_log = a_dis.log_prob(a_).detach().cpu().numpy()
        else:
            a_ = mu
            a_log = None

        if re_flag == True:
            a_ = rearrange(a_, '(b t) d -> b t d', b=b)
            a_log = rearrange(a_log, '(b t) d -> b t d', b=b)
        return a_.detach().cpu().numpy(), a_log

    def getActionDistribution(self, obs):
        pass

    def loss(self):
        pass
        # calculate reconstruction loss

        #recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

class CMT(nn.Module):
    def __init__(
        self,
        config,
        decoder_dim_head = 64
    ):
        super().__init__()
        # construct encoder
        self.config = config
        self.n_agent = config.block_size # self.n_agent means the max input block
        self.action_dim = config.action_dim
        self.obs_dim = config.obs_dim

        self.time_step = config.n_agent
        self.device = config.device

        self.embed_dim = config.embed_dim
        self.encoder_head = config.encoder_head
        self.encoder_dim = config.encoder_dim
        self.encoder_layer = config.encoder_layer
        self.decoder_head = config.decoder_head
        self.decoder_dim = config.decoder_dim
        self.decoder_layer = config.decoder_layer
        #self.mode = config.mode #'oa' 'oar' 'oaro'
        self.encoder = Transformer(self.encoder_dim,
                                   self.encoder_layer,
                                   self.encoder_head,
                                   decoder_dim_head,
                                   self.encoder_dim * 4) # TODO:dropout, dim_head

        # token [TODO: all the same or different?]
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.obs_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.action_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.reward_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.obs_next_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # position embeding
        self.obs_pos_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.action_pos_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.reward_pos_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        #self.obs_next_pos_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.agent_pos_embedding = nn.Parameter(torch.randn(1, self.n_agent, self.embed_dim))
        self.timestep_embeding = nn.Embedding(self.time_step, self.embed_dim)

        # embed [TODO: is ReLu necessary?]
        self.to_obs_embed = nn.Sequential(
            nn.Linear(self.obs_dim, self.embed_dim)
        )
        self.to_action_embed = nn.Sequential(
            nn.Linear(self.action_dim, self.embed_dim)
        )
        self.to_reward_embed = nn.Sequential(
            nn.Linear(1, self.embed_dim)
        )
        self.to_obs_next_embed = nn.Sequential(
            nn.Linear(self.obs_dim, self.embed_dim)
        )

        # decoder parameters
        self.enc_to_dec = nn.Linear(self.encoder_dim, self.decoder_dim) if self.encoder_dim != self.decoder_dim else nn.Identity()
        # [TODO: consider set the maksed input of decoder as masked token]
        self.decoder = maskedTransformer(dim = self.decoder_dim, depth = self.decoder_layer, heads = self.decoder_head, dim_head = self.decoder_dim, mlp_dim = self.decoder_dim * 4)

        # [TODO: should we set learning parameter for decoder]
        self.decoder_agent_pos_emb = nn.Embedding(self.n_agent, self.decoder_dim)
        self.decoder_obs_pos_emb = nn.Embedding(self.obs_dim, self.decoder_dim)
        self.decoder_action_pos_emb = nn.Embedding(self.action_dim, self.decoder_dim)
        self.decoder_reward_pos_emb = nn.Embedding(1, self.decoder_dim)

        self.embed_to_obs = nn.Sequential(
            nn.Linear(self.embed_dim, self.obs_dim)
        )
        self.embed_to_action = nn.Sequential(
            nn.Linear(self.embed_dim, self.action_dim)
        )
        self.embed_to_reward = nn.Sequential(
            nn.Linear(self.embed_dim, 1)
        )
        self.embed_to_obs_next = nn.Sequential(
            nn.Linear(self.embed_dim, self.obs_dim)
        )

        self.optimizer = self.configure_optimizers()

        self.parameter_number = sum(p.numel() for p in self.parameters())
        print("number of parameters: %e", self.parameter_number)


    def forward(self, obs=None, action=None, reward=None, futrue= False, train=False):
        """
        :param obs: [batch * n_agent/n_timestep * dim]
        :param action:
        :param reward:
        :param obs_next:
        :return: o,a,r after reconstruction
        """
        device = obs.device
        batch_num, n_agent, _ = obs.shape

        # patch to encoder tokens and add positions
        agent_pos_embedding = repeat(self.agent_pos_embedding, 'b n d -> (b b_repeat) (n n_repeat) d',
                                     b_repeat=batch_num, n_repeat=n_agent // self.n_agent + 1)
        agent_pos_embedding = agent_pos_embedding[:, :n_agent]

        # if mask, should use the specific token to replace the origin input
        # o-observation; a-action; r-reward; x-observation_next; g-agent

        action_token = self.to_action_embed(action) if action != None else 0
        obs_token = self.to_obs_embed(obs)  if obs != None else 0
        reward_token = self.to_reward_embed(reward) if reward != None else 0

        # TODO: may choose an more elegant method
        action_token[:,-1:] = repeat(self.action_token, 'b t d -> (b b_repeat) t d', b_repeat=batch_num)
        reward_token[:,-1:] = repeat(self.reward_token, 'b t d -> (b b_repeat) t d', b_repeat=batch_num)

        # TODO: check weather agent_postion_embeding is neccessary
        obs_token += self.obs_pos_embedding + agent_pos_embedding
        action_token += self.action_pos_embedding + agent_pos_embedding
        reward_token += self.reward_pos_embedding + agent_pos_embedding
        cls_token = repeat(self.cls_token, 'b t d -> (b b_repeat) t d', b_repeat=batch_num)

        # if input the current info or the info should as a results, should feed into the network.
        tokens_ls = [obs_token, action_token, reward_token]
        tokens_ls = rearrange(tokens_ls, 'n b t d-> b (t n) d') # change the seq [oooorrrrraaaa] to seq [oraoraoraora]
        tokens_ls = [cls_token, tokens_ls]
        tokens = torch.cat(tokens_ls, dim=1)

        # get the patches to be masked for the final reconstruction loss
        # attend with vision transformer [TODO: in CV, mlp head is used to merge infomation]

        encoded_tokens = self.encoder(tokens)

        # test:
        encoded_tokens[:, 0:1, :] = repeat(self.obs_next_token, 'b t d -> (b b_repeat) t d', b_repeat=batch_num)

        # concat context and the raw inputs
        autoregressive_tokens = torch.clone(tokens)
        autoregressive_tokens[:,0:1,:] = encoded_tokens[:,0:1,:]
        # TODO: check if neccessary
        decoder_tokens = self.enc_to_dec(autoregressive_tokens)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        # concat the masked tokens to the decoder tokens and attend with decoder
        if train == False:
            decoded_tokens = self.decoder(decoder_tokens)
        else:
            # the past context and last state
            decoder_tokens = encoded_tokens[:,0:1,:] + encoded_tokens[:, -3, :]
            for i in range(self.n_agent):
                decoded_tokens = self.decoder(decoder_tokens)
            pass

        # splice out the mask tokens and project to pixel values
        ## reconstruction
        # TODO: check drop the first context or the last placeholder?
        obs_tokens, reward_tokens, action_tokens = rearrange(decoded_tokens[:,1:,:], 'b (t n) d -> n b t d', n=3)

        obs = self.embed_to_obs(obs_tokens)
        action = self.embed_to_action(action_tokens)
        reward = self.embed_to_reward(reward_tokens)

        return obs, action, reward, encoded_tokens[:, 0:1, :]

    def mask(self, s, r, a, s_next, type=None):
        if type == 'action_full':
            mask_s = torch.zeros(s.shape)
            a = repeat(self.action_token, 'b n d -> (b b_repeat) (n n_repeat) d', b_repeat=a.shape[0], n_repeat=a.shape[1])
            mask_a = torch.ones(a.shape)
            return s, a, mask_s, mask_a


    def configure_optimizers(self):
        config = self.config.config
        #TODO: update optimaizer setting
        # learning rate schedular
        optimizer = torch.optim.AdamW(self.parameters(), lr=config['train']['lr'])#, betas=train_config.betas)
        return optimizer

    def getAction(self, obs=None, action=None, reward=None):
        # input dim [time * dim]
        obs = obs[-self.n_agent:,:].unsqueeze(0)
        action = action[-self.n_agent:,:].unsqueeze(0)
        reward = reward.reshape(1,-1,1)[:, -self.n_agent:, :]

        s, a, r, context = self.forward(obs=obs, action=action, reward=reward)
        a_ = a[0, -1, :].cpu().detach().numpy()
        return a_

    def getVecAction(self, obs=None, action=None, reward=None):
        # input dim [time * dim]
        obs = obs[:,-self.n_agent:, :].to(device=self.device)
        action = action[:,-self.n_agent:, :].to(device=self.device)
        reward = reward[:, -self.n_agent:, :].to(device=self.device)

        s, a, r, context = self.forward(obs=obs, action=action, reward=reward)
        a_ = a[:, -1, :].cpu().detach().numpy()
        return a_

    def getActionDistribution(self, obs):
        pass

    def loss(self):
        pass
        # calculate reconstruction loss

        #recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

class MAE_fewshot(nn.Module):
    def __init__(self, actor, eval_envs):
        super().__init__()
        self.in_map_n_agent = 8
        self.out_map_n_agent = 8

        self.action_dim = actor.action_dim
        self.obs_dim = actor.obs_dim
        # Upper case for the generalization representation
        # Lower case for the origin representation
        self.S2s_ls = nn.Linear(self.in_map_n_agent*actor.obs_dim, self.out_map_n_agent*actor.obs_dim)
        self.s2S_ls = nn.Linear(self.out_map_n_agent*actor.obs_dim, self.in_map_n_agent*actor.obs_dim)
        self.A2a_ls = nn.Linear(self.in_map_n_agent*actor.action_dim, self.out_map_n_agent*actor.action_dim)
        self.a2A_ls = nn.Linear(self.out_map_n_agent * actor.action_dim, self.in_map_n_agent * actor.action_dim)
        self.init_indentify([self.S2s_ls, self.s2S_ls, self.A2a_ls, self.a2A_ls])

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=actor.config.config['offline']['lr'])
        self.actor = actor
        self.device = actor.device

    def init_indentify(self, layer_ls):
        for layer in layer_ls:
            torch.nn.init.eye_(layer.weight)
            torch.nn.init.zeros_(layer.bias)



    def forward(self, obs=None, action=None, reward=None, obs_next=None, mask=None, rate=0.5):
        # input: batch * n_agent * dim
        batch, n_agent, dim = obs.shape
        if obs != None:
            obs = rearrange(obs, 'b n d -> b (n d)')
            obs = self.S2s_ls(obs)
            obs = rearrange(obs, 'b (n d) -> b n d', n=self.out_map_n_agent)
        if action !=None:
            action = rearrange(action, 'b n d -> b (n d)')
            action = self.A2a_ls(action)
            action = rearrange(action, 'b (n d) -> b n d', n=self.out_map_n_agent)

        #inference
        c_o, c_a, c_r, c_x = self.actor(obs, action, reward, obs_next, mask)

        if c_o != None:
            c_o = rearrange(c_o, 'b n d -> b (n d)')
            o = self.s2S_ls(c_o)
            o = rearrange(o, 'b (n d) -> b n d', n=n_agent)
        else:
            o = None
        if c_a !=None:
            c_a = rearrange(c_a, 'b n d -> b (n d)')
            a = self.a2A_ls(c_a)
            a = rearrange(a, 'b (n d) -> b n d', n=n_agent)
        else:
            a = None
        if c_x !=None:
            c_x = rearrange(c_x, 'b n d -> b (n d)')
            x = self.s2S_ls(c_x)
            x = rearrange(x, 'b (n d) -> b n d', n=n_agent)
        else:
            x = None

        return o, a, None, x





