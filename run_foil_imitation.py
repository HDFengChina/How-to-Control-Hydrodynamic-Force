from env.flow_field_env import fake_env, foil_env
import argparse
import json
from model.online_gpt_model import GPTConfig, GPT
from framework.utils import set_seed, ConfigDict, make_logpath
from framework.logger import LogServer, LogClient
# from framework.buffer import OnlineBuffer
from framework.trainer1 import RSAC #, Trainer, SAC
from framework import utils
from model.sin_policy import SinPolicy
from datetime import datetime, timedelta
import wandb
import numpy as np
from tqdm import tqdm
import gym
import pickle


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
    args.name = f"[Debug]foil_newEnvV6_rnnUpdateValue_fixSacLogp_BC_{args.seed}"
    # v4 actionDevide_eta_actionRange

    # wandb remote logger, can mute when debug
    mute = True
    remote_logger = LogServer(args, mute=mute)  # TODO: open logging when finish debugging
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
dir = "config/rsac.yaml"
config_dict = utils.load_config(dir)
paras = utils.get_paras_from_dict(config_dict)
print("local:", paras)
wandb.init(project="foil", config=paras, name=args.name)
paras = utils.get_paras_from_dict(wandb.config)
print("finetune", paras)
run_dir, log_dir = make_logpath('foil', paras.algo)
### start env
env = foil_env(paras)
# env = gym.vector.make('foil-v0', num_envs=3)
paras.action_space, action_dim = env.action_dim, env.action_dim
paras.obs_space, observation_dim = env.observation_dim, env.observation_dim

env.reset()


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
episode_name = '[AD]0.3[St]0.1[theta]45[Phi]90_online_new'
offline_data = pickle.load(open(f"/mnt/nasdata/runji/Lilypad_new/offline_data/{episode_name}.pkl", "rb"))

agent = RSAC(paras)
agent.load('/mnt/nasdata/runji/LilyPad/models/foil/rsac/run197/trained_model/actor_offline.pth')
# imitation learning
agent.imitation_train(offline_data, iter=1500)
# save agent model
agent.save(run_dir, 'offline')

ret = []
obs = env.reset()
Gt = 0
done = False
epsoide_length = 0
epsoide_num = 0

buffer_new_data = 0
for i in tqdm(1):
    # rollout in env  -  rollout()
    agent.reset_rnn()
    ct_sum, eta_sum, cp_sum, fx_sum = 0, 0, 0, 0
    ct_ls, cp_ls, fx_ls, dt_ls = [], [], [], []
    while not done:
        action = agent.choose_action(obs)
        # print("==================action:", action, "==================================")
        # print("===============================", type(obs), obs, "==========================================")
        next_obs, reward, done, info = env.step(action["action"][0])
        # print("==================reward:", reward, "==================================")
        # if np.abs(action["action"][0][0]) > 1.2:
        #     reward -= 10000000
        # if np.abs(action["action"][0][1]) > 1.5:
        #     reward -= 10000000
        # time_weight = paras.time_weight
        # y_weight = paras.y_weight
        # alpha_weight = paras.alpha_weight
        # reward_shaped = reward + time_weight  #shaped reward
        # if np.abs(obs.squeeze()[0])>10:
        #    reward_shaped -= y_weight * np.abs(obs.squeeze()[0])
        # if np.abs(obs.squeeze()[1])>10:
        #     reward_shaped -= np.abs(obs.squeeze()[1])
        agent.add_experience(
            {"states": np.array(obs).astype(float).reshape(1, observation_dim),
             "states_next": np.array(next_obs).astype(float).reshape(1, observation_dim), "rewards": reward,
             "dones": np.float32(done)})
        obs = next_obs
        Gt += reward
        ct_sum += info['ct']
        eta_sum += info['eta']
        cp_sum += info['cp']
        fx_sum += info['fx']
        cp_ls.append(info['cp'])
        fx_ls.append(info['fx'])
        ct_ls.append(info['ct'])
        dt_ls.append(info['dt'])
        epsoide_length += 1
        buffer_new_data += 1
        if buffer_new_data > paras.batch_size:
            # it is better to use data for only 1-2 times
            agent.learn()
            buffer_new_data = 0
            wandb.log({"reward": reward, "critic_loss": agent.c_loss, "actor_loss": agent.a_loss,
                       "alpha_loss": agent.alpha_loss})
    obs = env.reset()
    done = False
    ret.append(Gt)
    avg_reward = (Gt - reward) / epsoide_length
    eff_avg_reward = avg_reward if epsoide_length > 8000 / paras.action_interval else 0  # set too short episode to zero
    # cal ct, eta, cp, fx from the middle
    ct_avg = np.average(ct_ls[int(epsoide_length / 2):], weights=dt_ls[int(epsoide_length / 2):])
    eta_avg = -1* np.average(fx_ls[int(epsoide_length / 2):], weights=dt_ls[int(epsoide_length / 2):]) / np.average(
        cp_ls[int(epsoide_length / 2):], weights=dt_ls[int(epsoide_length / 2):])
    cp_avg = np.mean(cp_ls)
    fx_avg = np.mean(fx_ls)
    wandb.log({"Gt": Gt, "length": epsoide_length, "episode_num": epsoide_num, "avg_reward": avg_reward,
               "effective_avg": eff_avg_reward, "ct": ct_avg, 'eta': eta_avg, "value": agent.value(obs),
               "policy_entropy": agent.ent, "alpha": agent.alpha.detach().cpu().numpy(),
               "time": np.sum(dt_ls)})  # avg should remove the final step reward, since it can be too large
    print("epoch:", i, "length:", epsoide_length, "G:", Gt, "actor_loss:", agent.a_loss, "critic_loss", agent.c_loss)
    print("action:", action, "; state: ", next_obs, "; reward:", reward)
    Gt = 0
    ct_sum, eta_sum, cp_sum, fx_sum = 0, 0, 0, 0
    ct_ls, cp_ls, fx_ls, dt_ls = [], [], [], []
    epsoide_length = 0
    epsoide_num += 1
    # update policy   -  update()
    # agent.learn()
    # eval policy performance - rollout(train=False)
    # logger
    # if i % 10 == 0:

    # save model parameter
    if i % 1000 == 0:
        agent.save(run_dir, i)
env.close()
