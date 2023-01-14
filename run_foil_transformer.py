from env.flow_field_env import fake_env, foil_env
import argparse
import json
from model.online_gpt_model import GPTConfig, GPT
from framework.utils import set_seed, ConfigDict, make_logpath
from framework.logger import LogServer, LogClient
# from framework.buffer import OnlineBuffer
from framework.trainer1 import RSAC, TPPO  # , Trainer, SAC
from framework import utils
from framework.normalization import RewardScaling, Normalization
from model.sin_policy import SinPolicy
from datetime import datetime, timedelta
import wandb
import numpy as np
from tqdm import tqdm
import gym
import pickle
import torch


def prepare_arguments():
    parser = argparse.ArgumentParser()
    # Required_parameter
    parser.add_argument("--config-file", "--cf", default="./config/config.json",
                        help="pointer to the configuration file of the experiment", type=str)
    args, unknown = parser.parse_known_args()
    args.config = json.load(open(args.config_file, 'r', encoding='utf-8'))
    print(args.config)

    ### set seed
    if args.config['seed'] == "none":
        args.config['seed'] = datetime.now().microsecond % 65536
        args.seed = args.config['seed']
    set_seed(args.seed)

    # reconfig some parameter
    args.name = f"[Debug]foil_newEnvV8_Fixdt_PPO_transformer_{args.seed}"
    # v4 actionDevide_eta_actionRange

    # wandb remote logger, can mute when debug
    mute = True
    remote_logger = LogServer(args, mute=mute)  # open logging when finish debuging
    remote_logger = LogClient(remote_logger)

    # for the hyperparameter search
    if mute:
        new_args = args
    else:
        new_args = remote_logger.server.logger.config if not mute else args
        new_args = ConfigDict(new_args)
        new_args.washDictChange()
    new_args.remote_logger = remote_logger
    return new_args, remote_logger


### load config AND prepare logger
args, remote_logger = prepare_arguments()
config = args.config
dir = "config/tppo.yaml"
config_dict = utils.load_config(dir)
paras = utils.get_paras_from_dict(config_dict)
run_dir, log_dir = make_logpath('foil', paras.algo)
print("local:", paras)
paras.run_dir = run_dir # logging for the model search
# wandb.init(project="foil", config=paras, name=args.name, mode="online")# ("disabled" or "online")
# paras = utils.get_paras_from_dict(wandb.config)
# print("finetune", paras)
### start env
#env = foil_env(paras)
num_envs = 10
env = gym.vector.make('foil-v0', num_envs=num_envs, config=paras)
env.reset()
#paras.action_space, action_dim = env.envs[0].action_dim, env.envs[0].action_dim
#paras.obs_space, observation_dim = env.envs[0].observation_dim, env.envs[0].observation_dim
paras.action_space, action_dim = env.single_action_space.shape[0], env.single_action_space.shape[0]
paras.obs_space, observation_dim = env.single_observation_space.shape[0], env.single_observation_space.shape[0]
paras.device = "cuda" if torch.cuda.is_available() else "cpu"
paras.env_num = num_envs
# action = [0,0]
# env.step(action)
# ### init poliy network
# mconf_critic = GPTConfig(observation_dim, action_dim, block_size=config['context_length'], config=config,
#                          embed_dim= args.config['model']['critic_n_embd'], encoder_dim=args.config['model']['critic_n_embd'], encoder_layer=args.config['model']['critic_n_layer'], encoder_head=args.config['model']['critic_n_head'],
#                          mode=config['mode'])

# ### init buffer
# buffer = OnlineBuffer(config['context_length'], observation_dim, action_dim, config)
# trainer = Trainer
### train

# agent = SAC(paras)

## offline data
# next_obs = env.reset()
# done = False
# epsoide_length = 0
# obs_ls, action_ls = [], []
# while not done:
#     action = ref_agent.choose_action(epsoide_length*0.005)
#     obs_ls.append(next_obs)
#     action_ls.append(action)
#     print(f"action [{action[0]}] [{action[1]}]")
#     next_obs, reward, done, info = env.step(action)
#     epsoide_length += 1

# load offline data from pickle
#episode_name = '[AD]0.9[St]0.1[theta]50[Phi]90_online'
#offline_data = pickle.load(open(f"/mnt/nasdata/runji/LilyPad/offline_data/{episode_name}.pkl", "rb"))
# since training on A100 and 3090, we need to align the data type when loading model
#torch.set_default_tensor_type(torch.DoubleTensor)
agent = TPPO(paras)
if paras.load_pretrained:
    agent.load('/mnt/nasdata/runji/Lilypad_new/models/foil/tppo/run60/trained_model/actor_2000.pth')
# imitation learning
# agent.imitation_train(offline_data, iter=1)
# save agent model
#agent.save(run_dir, 'offline')
# agent.load('/mnt/nasdata/runji/LilyPad/models/foil/rsac/run194/trained_model/actor_0.pth')



# sample episode in vector env

# add info into buffer

# sample batch from buffer, train agent
state_norm_flag = False
state_norm = Normalization((num_envs, paras.obs_space))
reward_norm = RewardScaling((num_envs), 0.99)
ret = []
obs = env.reset()
obs = state_norm(obs) if state_norm_flag else obs
done = [False] * num_envs
epsoide_length = 0
epsoide_num = 0
buffer_new_data = 0
Gt_best = -100000
reference_agent = SinPolicy(paras)
agent.reset_optimizer()  # change the mode from offline to online
for i in tqdm(range(config['epochs'])):
    # rollout in env  -  rollout()
    epsoide_length, Gt, time_cost = 0, 0, 0
    ct_ls, cp_ls, fx_ls, dt_ls = [], [], [], []
    agent.reset_state()
    info = [{}]
    info[0]['dt'] = 0
    while not any(done) and epsoide_length < 800:
        if epsoide_num == 0 and paras.load_pretrained == False:
            # off-policy sample from expert, and get enough cuda memory
            action = agent.choose_action(obs)
            action = reference_agent.choose_action(time_cost, dt=info[0]['dt'])
            # TODO: change into multi-reference agent
            action = np.tile(action, [num_envs, 1])
        else:
            action = agent.choose_action(obs)
        next_obs, ori_reward, done, info = env.step(action)
        # [trick] normalization of reward and observation
        next_obs = state_norm(next_obs) if state_norm_flag else next_obs
        reward = reward_norm(ori_reward)#ori_reward#reward_norm(ori_reward)
        # save to buffer
        next_state = next_obs.reshape(next_obs.shape[0], 1, next_obs.shape[1])
        next_state = np.concatenate([agent.state.cpu().detach().numpy(), next_state], axis=1)[:, 1:, :]
        agent.insert_data({'states': agent.state.cpu().detach().numpy(), 'actions': action, 'rewards': reward, 'states_next': next_state, 'dones': done})
        obs = next_obs
        Gt += ori_reward[0]
        cp_ls.append(info[0]['cp'])
        fx_ls.append(info[0]['fx'])
        ct_ls.append(info[0]['ct'])
        dt_ls.append(info[0]['dt'])
        epsoide_length += 1
        buffer_new_data += num_envs
        if buffer_new_data > paras.batch_size:
            # it is better to use data for only 1-2 times
            agent.learn()
            agent.memory.buffer_dict_clear()
            buffer_new_data = 0
            #wandb.log({"reward": reward, "critic_loss": agent.c_loss, "actor_loss": agent.a_loss, "alpha_loss": agent.alpha_loss})
    obs = env.reset()
    obs = state_norm(obs) if state_norm_flag else obs
    if buffer_new_data > 0:
        agent.learn()
        # clear online buffer
        agent.memory.buffer_dict_clear()
    done = [False] * num_envs
    ret.append(Gt)
    avg_reward = (Gt - reward[0]) / epsoide_length
    eff_avg_reward = avg_reward if epsoide_length > 8000 / paras.action_interval else 0  # set too short episode to zero
    # cal ct, eta, cp, fx from the middle
    ct_avg = np.average(ct_ls[int(epsoide_length / 2):], weights=dt_ls[int(epsoide_length / 2):])
    eta_avg = -1 * np.average(fx_ls[int(epsoide_length / 2):], weights=dt_ls[int(epsoide_length / 2):]) / np.average(cp_ls[int(epsoide_length / 2):], weights=dt_ls[int(epsoide_length / 2):])
    cp_avg = np.mean(cp_ls)
    fx_avg = np.mean(fx_ls)
    # wandb.log({"Gt": Gt, "length": epsoide_length, "episode_num": epsoide_num, "avg_reward": avg_reward,
    #            "effective_avg": eff_avg_reward, "ct": ct_avg, 'eta': eta_avg, "policy_entropy": agent.actor.entropy, "clip_frac": agent.clipfrac, "approxkl": agent.approxkl,
    #             "time": np.sum(dt_ls), "reward": reward, "critic_loss": agent.critic_loss, "actor_loss": agent.actor_loss, "action": np.array(action[0])})  # avg should remove the final step reward, since it can be too large
    print("epoch:", i, "length:", epsoide_length, "G:", Gt, "actor_loss:", agent.actor_loss, "critic_loss", agent.critic_loss, " sigma:", agent.actor.sigma_param)
    print("action:", action)# "; state: ", next_obs, "; reward:", reward)
    epsoide_num += 1
    # update policy   -  update()
    # agent.learn()
    # eval policy performance - rollout(train=False)
    # logger
    # if i % 10 == 0:

    # save model parameter
    if Gt > Gt_best and epsoide_num > 10:
        Gt_best = Gt
        agent.save(run_dir, "best")
        print("save best model")
    if i % 200 == 0:
        agent.save(run_dir, i)
env.__del__()
env.terminate()
