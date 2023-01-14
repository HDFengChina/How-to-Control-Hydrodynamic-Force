from env.flow_field_env import foil_env
import argparse
import json
from model.online_gpt_model import GPTConfig, GPT
from framework.utils import set_seed, ConfigDict, make_logpath
# from framework.logger import LogServer, LogClient
# from framework.buffer import OnlineBuffer
# from framework.trainer1 import RSAC, TPPO  # , Trainer, SAC
from framework import utils
from framework.normalization import RewardScaling, Normalization
from model.sin_policy import SinPolicy
from datetime import datetime, timedelta
# import wandb
import numpy as np
from tqdm import tqdm
from agent_TD3 import Agent
from server_agent import Server
import gym
import pickle
import torch

def prepare_arguments():
    parser = argparse.ArgumentParser(description="server for experiments and CFD")
    parser.add_argument("-env", "--environment", choices=["CFD", "Experiment"], default="Experiment")
    parser.add_argument("-fil", "--filter", choices=["None", "Kalman"], default="Kalman")
    parser.add_argument("-exn", "--explore_noise", choices=["Gaussian", "Process"], default="Gaussian")
    parser.add_argument("-one", "--one_action", action='store_true')
    parser.add_argument("--config-file", "--cf", default="./config/config.json",
                        help="pointer to the configuration file of the experiment", type=str)  # 参数被储存在config.json里
    args = parser.parse_args()
    args.config = json.load(open(args.config_file, 'r', encoding='utf-8'))
    print(args.config)

    ### set seed
    # if args.config['seed'] == "none":
    #     args.config['seed'] = datetime.now().microsecond % 65536
    #     args.seed = args.config['seed']
    # set_seed(args.seed)  # 定义随机种子为当前时刻毫秒

    # reconfig some parameter
    # args.name = f"[Debug]gym_PPO_transformer_{args.seed}"
    # v4 actionDevide_eta_actionRange

    # wandb remote logger, can mute when debug
    mute = True
    # remote_logger = LogServer(args, mute=mute)  # open logging when finish debuging
    # remote_logger = LogClient(remote_logger)

    # for the hyperparameter search
    # if mute:
    #     new_args = args
    #else:
    #    new_args = remote_logger.server.logger.config if not mute else args
    #    new_args = ConfigDict(new_args)
    #    new_args.washDictChange()
    #new_args.remote_logger = remote_logger
    return args

### load config AND prepare logger
args = prepare_arguments()
config = args.config
dir = "config/tppo_2.yaml"
config_dict = utils.load_config(dir)  # TPPO参数配置
paras = utils.get_paras_from_dict(config_dict)
print("local:", paras)
# wandb.init(project="foil", config=paras, name=args.name, mode="disabled")# ("disabled" or "online")
# paras = utils.get_paras_from_dict(wandb.config)  # 如果是disabled与上面paras一样
# print("finetune", paras)
run_dir, log_dir = make_logpath('foil', paras.algo)  # 设置actor模型保存地址

### start env
num_envs = 1  # 并行运算数量
env = gym.vector.make('foil-v0', num_envs=num_envs, config=paras)
# env.reset()
action_space, action_dim = env.single_action_space.shape[0], env.single_action_space.shape[0]  # 3,3

obs_space, observation_dim = env.single_observation_space.shape[0], env.single_observation_space.shape[0]  # 2, 2

# paras.device = "cuda" if torch.cuda.is_available() else "cpu"
# paras.env_num = num_envs

### train
# sample batch from buffer, train agent
state_norm_flag = False
state_norm = Normalization((num_envs, obs_space))
# reward_norm = RewardScaling(1, 0.99)
ret = []
obs = env.reset()
# obs = state_norm(obs) if state_norm_flag else obs  # 当flag=True归一化，False则不变
done = [False] * num_envs  # 定义done为非
epsoide_length = 0
epsoide_num = 0
buffer_new_data = 0
agent = Server(**args.__dict__)  # 此处定义server时情况一下buffer
agent._init(-1)  # 此处清空一下buffer, 这里与本地一致


for i in tqdm(range(config['epochs'])):  # epochs=4000
    agent._start_episode(-1)
    # rollout in env  -  rollout()
    epsoide_length, Gt = 0, 0
    ct_ls, cp_ls, fx_ls, dt_ls = [], [], [], []
    reward_buffer = []
    
    while not any(done):
        
        print(obs)
        action = agent._request_stochastic_action(obs)
        print(action)
        next_obs, ori_reward, done, info = env.step(action)
        print("Get next state", epsoide_length)
        # [trick] normalization of reward and observation
        next_obs = state_norm(next_obs) if state_norm_flag else next_obs
        reward_buffer.append(ori_reward)
        reward = ori_reward
        # save to buffer
        # next_state = next_obs.reshape(next_obs.shape[0], 1, next_obs.shape[1])  # 从二维变成三维数组

        # 将s,a存入buffer
        # agent.insert_data({'states': agent.state.cpu().detach().numpy(), 'actions': action, 'rewards': reward, 'states_next': next_state, 'dones': done})
        obs = next_obs
        Gt += ori_reward[0]
        epsoide_length += 1
        buffer_new_data += num_envs  # buffer里存了多少size
        if buffer_new_data > paras.batch_size:
            # it is better to use data for only 1-2 times
            #agent.learn()
            buffer_new_data = 0
            #wandb.log({"reward": reward, "critic_loss": agent.c_loss, "actor_loss": agent.a_loss, "alpha_loss": agent.alpha_loss})
    obs = env.reset()
    # print("Reset")
    reward_eps = np.average(reward_buffer)
    # print(reward_eps)  # 不作为优化只是查看效果使用
    obs = state_norm(obs) if state_norm_flag else obs
    agent._train(1000)  # 在此处往buffer存数据
    print("A time train end", i)
    # clear online buffer
    # agent.memory.buffer_dict_clear()
    done = [False] * num_envs
    ret.append(Gt)
    # avg_reward = (Gt - reward[0]) / epsoide_length
    # eff_avg_reward = avg_reward if epsoide_length > 8000 / paras.action_interval else 0  # set too short episode to zero
    # cal ct, eta, cp, fx from the middle
    # wandb.log({"Gt": Gt, "length": epsoide_length, "episode_num": epsoide_num, "avg_reward": avg_reward,
    #            "effective_avg": eff_avg_reward,"policy_entropy": agent.actor.entropy, "clip_frac": agent.clipfrac, "approxkl": agent.approxkl,
    #             "time": np.sum(dt_ls), "reward": reward, "critic_loss": agent.critic_loss, "actor_loss": agent.actor_loss})  # avg should remove the final step reward, since it can be too large
    # print("epoch:", i, "length:", epsoide_length, "G:", Gt, "actor_loss:", agent.actor_loss, "critic_loss", agent.critic_loss, " sigma:", agent.actor.sigma_param)
    # print("action:", action, "; state: ", next_obs, "; reward:", reward)
    epsoide_num += 1
    # update policy   -  update()
    # agent.learn()
    # eval policy performance - rollout(train=False)
    # logger
    # if i % 10 == 0:

    # save model parameter
    agent._save()

env.terminate()
