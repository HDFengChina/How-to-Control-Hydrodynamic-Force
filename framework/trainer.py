"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import copy
import math
from operator import imod

import wandb
from matplotlib.image import imread
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.distributions import Categorical
import torch.nn as nn
import os
from einops import rearrange, repeat
from torch.optim import Adam
from .buffer import Replay_buffer as buffer
from model.actor import GaussianActor, RActor
from model.critic import Critic, RCritic


class TrainerConfig:
    # optimization parameters
    max_epochs = 1
    grad_norm_clip = 0.5

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class SAC(object):
    def weights_init_(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

    def update_params(optim, loss, clip=False, param_list=False,retain_graph=False):
        optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if clip is not False:
            for i in param_list:
                torch.nn.utils.clip_grad_norm_(i, clip)
        optim.step()
    def __init__(self, args):


        def get_trajectory_property():  #for adding terms to the memory buffer
            return ["action"]


        def weights_init_(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                torch.nn.init.constant_(m.bias, 0)

        self.state_dim = args.obs_space
        self.action_dim = args.action_space

        self.gamma = args.gamma
        self.tau = args.tau

        self.action_continuous = args.action_continuous

        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.actor_lr = args.a_lr
        self.critic_lr = args.c_lr
        self.alpha_lr = args.alpha_lr
        self.c_loss = 0
        self.a_loss = 0

        self.buffer_size = args.buffer_capacity

        self.policy_type = 'discrete' if (not self.action_continuous) else args.policy_type      #deterministic or gaussian policy
        self.device = 'cuda'

        given_critic = Critic   #need to set a default value
        self.preset_alpha = args.alpha

        self.tune_entropy = args.tune_entropy
        self.target_entropy_ratio = args.target_entropy_ratio

        self.policy = GaussianActor(self.state_dim, self.hidden_size, 2, tanh = True, action_high = args.action_max, action_low = -args.action_max).to(self.device)
        #self.policy_target = GaussianActor(self.state_dim, self.hidden_size, 1, tanh = False).to(self.device)

        hid_layer = args.num_hid_layer
        self.q1 = given_critic(self.state_dim+self.action_dim, self.action_dim, self.hidden_size, hid_layer).to(self.device)
        self.q1.apply(weights_init_)
        self.critic_optim = Adam(self.q1.parameters(), lr = self.critic_lr)

        self.q1_target = given_critic(self.state_dim+self.action_dim, self.action_dim, self.hidden_size, hid_layer).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.learn_step_counter = 0
        self.target_replace_iter = args.target_replace

        self.policy_optim = Adam(self.policy.parameters(), lr = self.actor_lr)
        # TODO buffer
        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()

        if self.tune_entropy:
            self.target_entropy = -np.log(1./self.action_dim) * self.target_entropy_ratio
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            #self.alpha = self.log_alpha.exp()
            self.alpha = torch.tensor([self.preset_alpha], device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = torch.tensor([self.preset_alpha], device=self.device)  # coefficiency for entropy term


    def choose_action(self, state, train = True):
        state = torch.tensor(state, dtype=torch.float, device=self.device).view(1, -1)
        if train:
            action, _, _ = self.policy.sample(state)
            action = action.detach().cpu().numpy()
            self.add_experience({"action": action})
        else:
            _, _, action = self.policy.sample(state)
            action = action.cpu().item()
        return {'action':action}



    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)


    def critic_loss(self, current_state, batch_action, next_state, reward, mask):

        with torch.no_grad():
            next_state_action, next_state_pi, next_state_log_pi, _ = self.policy.sample(next_state)
            #qf1_next_target, qf2_next_target = self.critic_target(next_state)
            qf1_next_target = self.q1_target(next_state)
            qf2_next_target = self.q2_target(next_state)

            min_qf_next_target = next_state_pi * (torch.min(qf1_next_target, qf2_next_target) - self.alpha
                                                  * next_state_log_pi)  # V function
            min_qf_next_target = min_qf_next_target.sum(dim=1, keepdim=True)
            next_q_value = reward + mask * self.gamma * (min_qf_next_target)

        #qf1, qf2 = self.critic(current_state)  # Two Q-functions to mitigate positive bias in the policy improvement step, [batch, action_num]
        qf1 = self.q1(current_state)
        qf2 = self.q2(current_state)

        qf1 = qf1.gather(1, batch_action.long())
        qf2 = qf2.gather(1, batch_action.long())        #[batch, 1] , pick the actin-value for the given batched actions

        qf1_loss = torch.mean((qf1 - next_q_value).pow(2))
        qf2_loss = torch.mean((qf2 - next_q_value).pow(2))

        return qf1_loss, qf2_loss

    def policy_loss(self, current_state):

        with torch.no_grad():
            #qf1_pi, qf2_pi = self.critic(current_state)
            qf1_pi = self.q1(current_state)
            qf2_pi = self.q2(current_state)

            min_qf_pi = torch.min(qf1_pi, qf2_pi)

        pi, prob, log_pi, _ = self.policy.sample(current_state)
        inside_term = self.alpha.detach() * log_pi - min_qf_pi  # [batch, action_dim]
        policy_loss = ((prob * inside_term).sum(1)).mean()

        return policy_loss, prob.detach(), log_pi.detach()

    def alpha_loss(self, action_prob, action_logprob):

        if self.tune_entropy:
            entropies = -torch.sum(action_prob * action_logprob, dim=1, keepdim=True)       #[batch, 1]
            entropies = entropies.detach()
            alpha_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropies))

            alpha_logs = self.log_alpha.exp().detach()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_logs = self.alpha.detach().clone()

        return alpha_loss, alpha_logs

    def learn(self):
        def update_params(optim, loss, clip=False, param_list=False,retain_graph=False):
            optim.zero_grad()
            loss.backward(retain_graph=retain_graph)
            if clip is not False:
                for i in param_list:
                    torch.nn.utils.clip_grad_norm_(i, clip)
            optim.step()

        data = self.memory.sample(self.batch_size)

        transitions = {
            "o_0": np.array(data['states']),
            "o_next_0": np.array(data['states_next']),
            "r_0": np.array(data['rewards']).reshape(-1, 1),
            "u_0": np.array(data['action']),
            "d_0": np.array(data['dones']).reshape(-1, 1),
        }

        obs = torch.tensor(transitions["o_0"], dtype=torch.float, device=self.device)
        obs_ = torch.tensor(transitions["o_next_0"], dtype=torch.float, device=self.device)
        action = torch.tensor(transitions["u_0"], dtype=torch.long, device=self.device).view(self.batch_size, -1)
        reward = torch.tensor(transitions["r_0"], dtype=torch.float, device=self.device)
        done = torch.tensor(transitions["d_0"], dtype=torch.float, device=self.device)

        action = torch.tensor(transitions["u_0"], dtype=torch.float, device=self.device).view(self.batch_size, -1)

        with torch.no_grad():
            # next_action, next_action_logprob, _ = self.policy_target.sample(obs_)
            next_action, next_action_logprob, _ = self.policy.sample(obs_)
            target_next_q = self.q1_target(
                torch.cat([obs_, next_action], -1)).squeeze() - (self.alpha * next_action_logprob).squeeze()
            next_q_value = reward + (1 - done) * self.gamma * target_next_q
        qf1 = self.q1(torch.cat([obs.squeeze(), action], -1))
        qf_loss = F.mse_loss(qf1, next_q_value)

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(obs)
        qf_pi = self.q1(torch.cat([obs.squeeze(), pi.squeeze()], -1))
        policy_loss = ((self.alpha * log_pi) - qf_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self.c_loss = qf_loss.detach().cpu().numpy()
        self.a_loss = policy_loss.detach().cpu().numpy()

        if self.tune_entropy:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        else:
            pass

        if self.learn_step_counter % self.target_replace_iter == 0:
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)
            # for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
            #    target_param.data.copy_(self.tau * param.data + (1.-self.tau) * target_param.data)

    def save(self, save_path, episode):

        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.policy.state_dict(), model_actor_path)

    def load(self, file):
        self.policy.load_state_dict(torch.load(file))


class RSAC(object):
    def weights_init_(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

    def update_params(optim, loss, clip=False, param_list=False,retain_graph=False):
        optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if clip is not False:
            for i in param_list:
                torch.nn.utils.clip_grad_norm_(i, clip)
        optim.step()
    def __init__(self, args):


        def get_trajectory_property():  #for adding terms to the memory buffer
            return ["action"]


        def weights_init_(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                torch.nn.init.constant_(m.bias, 0)

        self.state_dim = args.obs_space
        self.action_dim = args.action_space

        self.gamma = args.gamma
        self.tau = args.tau

        self.action_continuous = args.action_continuous

        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.actor_lr = args.a_lr
        self.critic_lr = args.c_lr
        self.alpha_lr = args.alpha_lr
        self.c_loss = 0
        self.a_loss = 0
        self.ent = 0

        self.buffer_size = args.buffer_capacity

        self.policy_type = 'discrete' if (not self.action_continuous) else args.policy_type      #deterministic or gaussian policy
        self.device = 'cuda'

        given_critic = RCritic   #need to set a default value
        self.preset_alpha = args.alpha

        self.tune_entropy = args.tune_entropy
        self.target_entropy_ratio = args.target_entropy_ratio

        # self.policy = GaussianActor(self.state_dim, self.hidden_size, 2, tanh = False).to(self.device)
        self.policy = RActor(self.state_dim, self.hidden_size, self.action_dim, tanh=False, action_high = args.action_max, action_low = -args.action_max).to(self.device)
        #self.policy_target = GaussianActor(self.state_dim, self.hidden_size, 1, tanh = False).to(self.device)

        hid_layer = args.num_hid_layer
        # TODO
        # self.q1 = given_critic(self.state_dim+self.action_dim, self.action_dim, self.hidden_size, hid_layer).to(self.device)
        self.q1 = given_critic(self.state_dim+self.action_dim, 1, self.hidden_size, hid_layer).to(self.device)
        self.q1.apply(weights_init_)
        self.critic_optim = Adam(self.q1.parameters(), lr = self.critic_lr)
        # TODO
        # self.q1_target = given_critic(self.state_dim+self.action_dim, self.action_dim, self.hidden_size, hid_layer).to(self.device)
        self.q1_target = given_critic(self.state_dim+self.action_dim, 1, self.hidden_size, hid_layer).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.learn_step_counter = 0
        self.target_replace_iter = args.target_replace

        self.policy_optim = Adam(self.policy.parameters(), lr = self.actor_lr)
        # TODO buffer
        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()

        if self.tune_entropy:
            self.target_entropy = -np.log(1./self.action_dim) * self.target_entropy_ratio
            #self.alpha = self.log_alpha.exp()
            self.alpha = torch.tensor([self.preset_alpha]).to(self.device)
            self.log_alpha = torch.tensor(torch.log(self.alpha).detach().cpu().numpy(), requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = torch.tensor([self.preset_alpha]).to(self.device)  # coefficiency for entropy term

    def reset_optim(self):
        self.critic_optim = Adam(self.q1.parameters(), lr = self.critic_lr)
        self.policy_optim = Adam(self.policy.parameters(), lr = self.actor_lr)
        if self.tune_entropy:
            self.alpha_optim = Adam([self.log_alpha], lr=self.alpha_lr)


    def choose_action(self, state, train = True):
        state = torch.tensor(state, dtype=torch.float).view(1, 1, -1).to(self.device)
        if train:# TODO: check dimension meanings: batch * agent_num * dim
            action, _, _, hidden, next_hidden = self.policy.sample(state)
            action = action.detach().squeeze(0).cpu().numpy()
            t_action = torch.tensor(action, dtype=torch.float).view(1, 1, -1).to(self.device)
            ## inference Q for hidden state
            self.q1(torch.cat([state, t_action], -1))
            self.q1_target(torch.cat([state, t_action], -1))
            hidden_q, next_hidden_q, hidden_q_target, next_hidden_q_target = self.q1.hidden_q.detach().squeeze(
                0).cpu().numpy(), self.q1.next_hidden_q.detach().squeeze(
                0).cpu().numpy(), self.q1_target.hidden_q.detach().squeeze(
                0).cpu().numpy(), self.q1_target.next_hidden_q.detach().squeeze(0).cpu().numpy()

            self.add_experience({"action": action, "hidden": hidden, "next_hidden": next_hidden, "hidden_q": hidden_q, "next_hidden_q": next_hidden_q, "hidden_q_target": hidden_q_target, "next_hidden_q_target": next_hidden_q_target})
        else:
            _, _, action = self.policy.sample(state)
            action = action.item()
        return {'action':action}



    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)


    def critic_loss(self, current_state, batch_action, next_state, reward, mask):

        with torch.no_grad():
            next_state_action, next_state_pi, next_state_log_pi, _ = self.policy.sample(next_state)
            #qf1_next_target, qf2_next_target = self.critic_target(next_state)
            qf1_next_target = self.q1_target(next_state)
            qf2_next_target = self.q2_target(next_state)

            min_qf_next_target = next_state_pi * (torch.min(qf1_next_target, qf2_next_target) - self.alpha
                                                  * next_state_log_pi)  # V function
            min_qf_next_target = min_qf_next_target.sum(dim=1, keepdim=True)
            next_q_value = reward + mask * self.gamma * (min_qf_next_target)

        #qf1, qf2 = self.critic(current_state)  # Two Q-functions to mitigate positive bias in the policy improvement step, [batch, action_num]
        qf1 = self.q1(current_state)
        qf2 = self.q2(current_state)

        qf1 = qf1.gather(1, batch_action.long())
        qf2 = qf2.gather(1, batch_action.long())        #[batch, 1] , pick the actin-value for the given batched actions

        qf1_loss = torch.mean((qf1 - next_q_value).pow(2))
        qf2_loss = torch.mean((qf2 - next_q_value).pow(2))

        return qf1_loss, qf2_loss

    def policy_loss(self, current_state):

        with torch.no_grad():
            #qf1_pi, qf2_pi = self.critic(current_state)
            qf1_pi = self.q1(current_state)
            qf2_pi = self.q2(current_state)

            min_qf_pi = torch.min(qf1_pi, qf2_pi)

        pi, prob, log_pi, _ = self.policy.sample(current_state)
        inside_term = self.alpha.detach() * log_pi - min_qf_pi  # [batch, action_dim]
        policy_loss = ((prob * inside_term).sum(1)).mean()

        return policy_loss, prob.detach(), log_pi.detach()

    def alpha_loss(self, action_prob, action_logprob):

        if self.tune_entropy:
            entropies = -torch.sum(action_prob * action_logprob, dim=1, keepdim=True)       #[batch, 1]
            entropies = entropies.detach()
            alpha_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropies))

            alpha_logs = self.log_alpha.exp().detach()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_logs = self.alpha.detach().clone()

        return alpha_loss, alpha_logs

    def reset_rnn(self):
        self.policy.set_h()
        self.q1.set_h()
        self.q1_target.set_h()

    def learn(self):
        def update_params(optim, loss, clip=False, param_list=False,retain_graph=False):
            optim.zero_grad()
            loss.backward(retain_graph=retain_graph)
            if clip is not False:
                for i in param_list:
                    torch.nn.utils.clip_grad_norm_(i, clip)
            optim.step()

        data = self.memory.sample2(self.batch_size)

        transitions = {
            "o_0": np.array(data['states']),
            "o_next_0": np.array(data['states_next']),
            "r_0": np.array(data['rewards']).reshape(-1, 1),
            "u_0": np.array(data['action']),
            "d_0": np.array(data['dones']).reshape(-1, 1),
        }

        obs = torch.tensor(transitions["o_0"], dtype=torch.float).view().to(self.device)
        obs_ = torch.tensor(transitions["o_next_0"], dtype=torch.float).to(self.device)
        action = torch.tensor(transitions["u_0"], dtype=torch.long).view(self.batch_size, -1).to(self.device)
        reward = torch.tensor(transitions["r_0"], dtype=torch.float).to(self.device)
        done = torch.tensor(transitions["d_0"], dtype=torch.float).to(self.device)
        hidden = torch.tensor(data["hidden"], dtype=torch.float).view(1, self.batch_size, -1).to(self.device) # timlens * batch *dim
        hidden_q = torch.tensor(data["hidden_q"], dtype=torch.float).view(1, self.batch_size, -1).to(self.device)
        hidden_q_target = torch.tensor(data["hidden_q_target"], dtype=torch.float).view(1, self.batch_size, -1).to(self.device)
        action = torch.tensor(transitions["u_0"], dtype=torch.float).view(self.batch_size, -1).to(self.device)

        # backup recent hidden state
        policy_hidden = self.policy.h
        q1_hidden = self.q1.h
        q1_target_hidden = self.q1_target.h

        # set hidden state in buffer
        self.policy.set_h(hidden)
        self.q1.set_h(hidden_q)
        self.q1_target.set_h(hidden_q_target)
        with torch.no_grad():
            # next_action, next_action_logprob, _ = self.policy_target.sample(obs_)
            next_action, next_action_logprob, _, _, _ = self.policy.sample(obs_)
            target_next_q = self.q1_target(
                torch.cat([obs_, next_action], -1)).squeeze() - (self.alpha * next_action_logprob).squeeze()
            next_q_value = reward.squeeze() + (1 - done.squeeze()) * self.gamma * target_next_q
        qf1 = self.q1(torch.cat([obs, action.view(-1, 1, 1)], -1)).squeeze() # TD-lambda
        qf_loss = F.mse_loss(qf1, next_q_value)

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        self.policy.set_h(hidden)
        self.q1.set_h(hidden_q)
        self.q1_target.set_h(hidden_q_target)
        pi, log_pi, _, _, _ = self.policy.sample(obs)
        qf_pi = self.q1(torch.cat([obs, pi], -1))
        policy_loss = ((self.alpha * log_pi) - qf_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self.c_loss = qf_loss.detach().cpu().numpy()
        self.a_loss = policy_loss.detach().cpu().numpy()
        entropies = -torch.sum(log_pi.exp() * log_pi, dim=1, keepdim=True)       #[batch, 1]
        entropies = entropies.detach().cpu().numpy().mean()
        self.ent = entropies

        if self.tune_entropy:
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_loss = alpha_loss.detach().cpu().numpy().mean()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().to(self.device)
        else:
            pass

        if self.learn_step_counter % self.target_replace_iter == 0:
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)
            # for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
            #    target_param.data.copy_(self.tau * param.data + (1.-self.tau) * target_param.data)

        # recover the hidden
        self.policy.set_h(policy_hidden)
        self.q1.set_h(q1_hidden)
        self.q1_target.set_h(q1_target_hidden)
        
    def value(self, state):
        self.q1_target.set_h()
        state = torch.tensor(state, dtype=torch.float).view(1, 1, -1).to(self.device)
        # action = torch.tensor(action).squeeze().to(self.device)
        action, _, _, _, _ = self.policy.sample(state)
        pair = torch.cat([state, action], dim=-1)
        # TODO
        value = self.q1_target(pair).cpu().detach().mean().numpy()
        return value

    def save(self, save_path, episode):

        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.policy.state_dict(), model_actor_path)

    def load(self, file):
        self.policy.load_state_dict(torch.load(file))

    def imitation_train(self, offline_data):
        action_set = offline_data[0]
        state_set = offline_data[1]
        #state_ls = [state_set['y'] / 30, state_set['theta'] / 50, state_set['yvel'] / 30, state_set['tvel'] / 30]
        # concatenate state list
        #state_ls = np.vstack(state_ls).T
        # convert to tensor
        state_ls = torch.tensor(state_set, dtype=torch.float).to(self.device)
        action = torch.tensor(action_set, dtype=torch.float).to(self.device)

        for iter in  range(10000):
            self.policy.set_h()
            policy_action_ls = []
            for i in range(len(action_set)):
                state = state_ls[i].view(1, 1, -1).to(self.device)
                next_action, next_action_logprob, _, _, _ = self.policy.sample(state)
                policy_action_ls.append(next_action)
            # concatenate policy action list
            policy_action = torch.cat(policy_action_ls, dim=0).squeeze()
            # cal the loss for action and policy_action
            loss = F.mse_loss(policy_action, action)
            self.imit_loss = loss.detach().cpu().numpy()
            self.policy_optim.zero_grad()
            loss.backward()
            self.policy_optim.step()
            if iter % 1 == 0:
                print("imitation train iter: ", iter, " loss: ", self.imit_loss)
                wandb.log({"imitation_train_loss": self.imit_loss})

        return loss.detach().cpu().numpy()



class Trainer:
    def __init__(self, actor, critic, config, args):
        self.actor = actor
        self.critic = critic
        self.config = config
        self.args = args

    def finetune(self, dataset):
        actor, critic, config = self.actor, self.critic, self.config
        if self.args.moe:
            target_actor = self.actor
        else:
            # target_model = copy.deepcopy(model)
            target_actor = self.actor
        target_actor.train(False)

        def run_epoch():
            actor.train(True)
            # critic.train(True)
            # loader = DataLoader(dataset, shuffle=True, pin_memory=True, drop_last=True,
            #                     batch_size=config.batch_size,
            #                     num_workers=config.num_workers)
            # loader = actor.deepspeed_io(dataset)
            # loader.data_sampler.set_epoch(epoch_idx)

            loss_info = 0
            mask_loss = {}
            pbar = tqdm(enumerate(dataset), total=int(len(dataset)))

            # todo: check these inputs
            # state, obs, action, reward, done, available_action
            # state, obs, action, reward, done, available_action, v_value, logp
            for it, (_,s, o, a, r, done, ava,_,_) in pbar:
                # place data on the correct device
                s_next = torch.tensor(s).unsqueeze(0).to(actor.device)[:, :, 1:, :]
                o_next =  torch.tensor(o).unsqueeze(0).to(actor.device)[:, :, 1:, :].float()
                s =  torch.tensor(s).unsqueeze(0).to(actor.device)[:, :, :-1, :]
                o =  torch.tensor(o).unsqueeze(0).to(actor.device)[:, :, :-1, :].float()
                a =  torch.tensor(a).unsqueeze(0).to(actor.device)[:, :, :-1, :]
                r =  torch.tensor(r).unsqueeze(0).to(actor.device)[:, :, :-1, :].float()
                ava =  torch.tensor(ava).unsqueeze(0).to(actor.device)[:, :, :-1, :]
                done =  torch.tensor(done).unsqueeze(0).unsqueeze(3).to(actor.device)

                # from [batch * agent * time* dim] into [(batch*time) * agent * dim]
                #s = rearrange(s, 'b n t d -> (b t) n d')
                o = rearrange(o, 'b n t d -> (b t) n d')
                o_next = rearrange(o_next, 'b n t d -> (b t) n d')
                a = rearrange(a, 'b n t d -> (b t) n d')
                r = rearrange(r, 'b n t d -> (b t) n d')

                a_onehot = torch.nn.functional.one_hot(a, num_classes=self.actor.action_dim).squeeze().float()
                # update actor
                with torch.set_grad_enabled(True):
                    if self.config.mode == "offline":
                        # mask data [mask and masked data]
                        # policy mask loss
                        #c_s, c_a, c_r, c_s_next = actor(o, mask="action_full")
                        #loss_a = F.cross_entropy(c_a.reshape(-1, c_a.shape[-1]), a.reshape(-1))

                        # agent mask loss[TODO: check if the method suitable?]
                        #c_s, c_a, c_r, c_s_next = actor(o, a_onehot, r, o_next, mask="g")
                        #loss_agent = F.cross_entropy(c_a.reshape(-1, c_a.shape[-1]), a.reshape(-1)) + F.mse_loss(
                        #    c_r.reshape(-1, c_r.shape[-1]), r.reshape(-1)) + F.mse_loss(c_s, o) + F.mse_loss(c_s_next,
                        #                                                                                     o_next)

                        # world dynamic loss[s,a -> s',r]
                        c_s, c_a, c_r, c_s_next = actor(o, a_onehot, mask="xr")
                        #loss_s = F.mse_loss(c_s, o)
                        o_next = rearrange(o_next, 'b n d -> b (n d)')
                        o_next_map = actor.S2s_ls(o_next)
                        o_next_map = rearrange(o_next_map, 'b (n d) -> b n d', n=o.shape[1])
                        loss_world = F.mse_loss(c_s_next, o_next_map)
                        #loss_reward = F.mse_loss(c_r.reshape(-1, c_r.shape[-1]), r.reshape(-1))

                        # opponent loss[s,s' -> a, r]
                        #c_s, c_a, c_r, c_s_next = actor(o, o_next, mask="ar")
                        #loss_opponent = F.cross_entropy(c_a.reshape(-1, c_a.shape[-1]), a.reshape(-1))

                        # masked_loss_a = loss_a[mask_for_a]

                        # [TODO: should search the mask coff parameter]
                        loss = loss_world #+ loss_s +



                        # loss_a + 1e-3 * (loss_agent + loss_world + loss_reward + loss_opponent + loss_s)

                        # actor.backward(loss)
                        actor.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), config.grad_norm_clip)
                        actor.optimizer.step()
                        # actor.step()

                        mask_loss = {"train_loss": loss.item(),
                                    # "policy_loss": loss_a.item(),
                                    # "agent_loss": loss_agent.item(),
                                     "dynamic_loss": loss_world.item(),
                                    # "reward_loss": loss_reward.item(),
                                    # "opponent_loss": loss_opponent.item(),
                                     #"representation_loss": loss_s.item()
                                     }

                        entropy_info = 0.
                        ratio_info = 0.
                        confidence_info = 0.
                        # report progress
                pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}.")
            return loss_info, None, entropy_info, ratio_info, confidence_info, mask_loss

        actor_loss_ret, critic_loss_ret, entropy, ratio, confidence = 0., 0., 0., 0., 0.
        for epoch in range(config.max_epochs):
            actor_loss_ret, critic_loss_ret, entropy, ratio, confidence, mask_info = run_epoch()
        return actor_loss_ret, critic_loss_ret, entropy, ratio, confidence, mask_info

    def train(self, dataset, epoch_idx=0, train_critic=True):
        actor, critic, config = self.actor, self.critic, self.config
        if self.args.moe:
            target_actor = self.actor
        else:
            # target_model = copy.deepcopy(model)
            target_actor = self.actor
        target_actor.train(False)

        def run_epoch():
            actor.train(True)
            #critic.train(True)
            loader = DataLoader(dataset, shuffle=True, pin_memory=True, drop_last=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
            #loader = actor.deepspeed_io(dataset)
            # loader.data_sampler.set_epoch(epoch_idx)

            loss_info = 0
            mask_loss = {}
            pbar = tqdm(enumerate(loader), total=int(len(loader)/loader.batch_size))

            # todo: check these inputs
            #state, obs, action, reward, done, available_action
            for it, (s, o, a, r, done, ava) in pbar:
                # place data on the correct device
                s_next = s.to(actor.device)[:, :, 1:, :]
                o_next = o.to(actor.device)[:, :, 1:, :].float()
                s = s.to(actor.device)[:, :, :-1, :]
                o = o.to(actor.device)[:, :, :-1, :]
                a = a.to(actor.device)[:, :, :-1, :]
                r = r.to(actor.device)[:, :, :-1, :].float()
                ava = ava.to(actor.device)[:, :, :-1, :]
                done = done.to(actor.device)



                # from [batch * agent * time* dim] into [(batch*time) * agent * dim]
                s = rearrange(s, 'b n t d -> (b t) n d')
                o = rearrange(o, 'b n t d -> (b t) n d')
                o_next = rearrange(o_next, 'b n t d -> (b t) n d')
                a = rearrange(a, 'b n t d -> (b t) n d')
                r = rearrange(r, 'b n t d -> (b t) n d')

                a_onehot= torch.nn.functional.one_hot(a, num_classes=self.actor.action_dim).squeeze().float()
                # update actor
                with torch.set_grad_enabled(True):
                    if self.config.mode == "offline":
                        # mask data [mask and masked data]
                        # policy mask loss
                        c_s, c_a, c_r, c_s_next = actor(o, mask="action_full")
                        loss_a = F.cross_entropy(c_a.reshape(-1, c_a.shape[-1]), a.reshape(-1))

                        # agent mask loss[TODO: check if the method suitable?]
                        c_s, c_a, c_r, c_s_next = actor(o, a_onehot, r, o_next, mask="g")
                        loss_agent =  F.cross_entropy(c_a.reshape(-1, c_a.shape[-1]), a.reshape(-1)) + F.mse_loss(c_r.reshape(-1), r.reshape(-1)) + F.mse_loss(c_s, o) + F.mse_loss(c_s_next, o_next)

                        # world dynamic loss[s,a -> s',r]
                        c_s, c_a, c_r, c_s_next = actor(o, a_onehot, mask="xr")
                        loss_s = F.mse_loss(c_s, o)
                        loss_world = F.mse_loss(c_s_next, o_next)
                        loss_reward = F.mse_loss(c_r.reshape(-1), r.reshape(-1))

                        # opponent loss[s,s' -> a, r]
                        c_s, c_a, c_r, c_s_next = actor(o, o_next, mask="ar")
                        loss_opponent = F.cross_entropy(c_a.reshape(-1, c_a.shape[-1]), a.reshape(-1))

                        #masked_loss_a = loss_a[mask_for_a]

                        # [TODO: should search the mask coff parameter]
                        loss = loss_a + 1e-3 * (loss_agent + loss_world + loss_reward + loss_opponent + loss_s)

                        mask_loss = {"train_loss": loss.item(),
                                     "policy_loss": loss_a.item(),
                                     "agent_loss": loss_agent.item(),
                                     "dynamic_loss": loss_world.item(),
                                     "reward_loss": loss_reward.item(),
                                     "opponent_loss": loss_opponent.item(),
                                     "representation_loss": loss_s.item()}

                        entropy_info = 0.
                        ratio_info = 0.
                        confidence_info = 0.
                    elif self.config.mode == "online":
                        logits = actor(o, pre_a, rtg, t)
                        adv = adv.reshape(-1, adv.size(-1))

                        logits[ava == 0] = -1e10
                        distri = Categorical(logits=logits.reshape(-1, logits.size(-1)))
                        target_a = a.reshape(-1)
                        log_a = distri.log_prob(target_a).unsqueeze(-1)

                        old_logits = target_actor(o, pre_a, rtg, t).detach()
                        old_logits[ava == 0] = -1e10
                        old_distri = Categorical(logits=old_logits.reshape(-1, old_logits.size(-1)))
                        old_log_a = old_distri.log_prob(target_a).unsqueeze(-1)

                        imp_weights = torch.exp(log_a - old_log_a)
                        actor_loss_ori = imp_weights * adv
                        actor_loss_clip = torch.clamp(imp_weights, 1.0 - 0.2, 1.0 + 0.2) * adv
                        actor_loss = -torch.min(actor_loss_ori, actor_loss_clip)
                        # actor_loss = -log_a * adv

                        act_entropy = distri.entropy().unsqueeze(-1)
                        loss = actor_loss - 0.001 * act_entropy
                        # loss = actor_loss

                        entropy_info = act_entropy.mean().item()
                        ratio_info = imp_weights.mean().item()
                        confidence_info = torch.exp(log_a).mean().item()
                    else:
                        raise NotImplementedError
                    loss = loss.mean()
                    loss_info = loss.item()

                #actor.backward(loss)
                actor.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), config.grad_norm_clip)
                actor.optimizer.step()
                #actor.step()

                # update critic
                critic_loss_info = 0.
                if train_critic:
                    #Todo:(Linghui) still no support for the critic model in DeepSpeed framework
                    with torch.set_grad_enabled(True):
                        v_value = critic(s, pre_a, rtg, t)
                        v_clip = v + (v_value - v).clamp(-0.2, 0.2)
                        critic_loss_ori = F.smooth_l1_loss(v_value.view(-1, 1), ret.view(-1, 1), beta=10)
                        critic_loss_clip = F.smooth_l1_loss(v_clip.view(-1, 1), ret.view(-1, 1), beta=10)
                        critic_loss = torch.max(critic_loss_ori, critic_loss_clip)

                        critic_loss_info = critic_loss.mean().item()

                    critic.backward(critic_loss)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), config.grad_norm_clip)
                    critic.step()

                # report progress
                pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}.")
            return loss_info, critic_loss_info, entropy_info, ratio_info, confidence_info, mask_loss

        actor_loss_ret, critic_loss_ret, entropy, ratio, confidence = 0., 0., 0., 0., 0.
        for epoch in range(config.max_epochs):
            actor_loss_ret, critic_loss_ret, entropy, ratio, confidence, mask_info = run_epoch()
        return actor_loss_ret, critic_loss_ret, entropy, ratio, confidence, mask_info

class FewShotLearner:
    def __init__(self, envs, config):
        self.envs = envs
        self.config = config

        self.s2x_ls = []
        self.x2s_ls = []
        for env in self.envs:
            pass
