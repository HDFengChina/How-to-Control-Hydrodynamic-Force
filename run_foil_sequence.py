from env.flow_field_env import fake_env, foil_env
import argparse
import json
from model.online_gpt_model import GPTConfig, GPT
from framework.utils import set_seed, ConfigDict, make_logpath
from framework.logger import LogServer, LogClient
# from framework.buffer import OnlineBuffer
#from framework.trainer import RSAC, Trainer, SAC
from framework import utils
from model.sin_policy import SinPolicy
from datetime import datetime, timedelta
import wandb
import numpy as np
from tqdm import tqdm
import csv
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
        args.config['seed'] = datetime.now().microsecond % 6553600
        args.seed = args.config['seed']
    set_seed(args.seed)

    # reconfig some parameter
    args.name = f"[Search]FixSinPolicy{args.seed}"

    # wandb remote logger, can mute when debug
    mute = True
    remote_logger = LogServer(args, mute=mute)  # TODO: open logging when finish debuging
    remote_logger = LogClient(remote_logger)

    # for the hyperparameter search
    if mute:
        new_args = args
    else:
        new_args = remote_logger.server.logger.config if not mute else args
        new_args = ConfigDict(new_args)
        new_args.washDictChange()
    new_args.remote_logger = remote_logger
    return new_args, remote_logger, unknown


### load config AND prepare logger
args, remote_logger, unknown = prepare_arguments()
config = args.config
dir = "config/sin.yaml"
config_dict = utils.load_config(dir)
paras = utils.get_paras_from_dict(config_dict)
print("local:", paras)
wandb.init(project="foil", config=paras, name=args.name, mode="disabled")# ("disabled" or "online")
paras = utils.get_paras_from_dict(wandb.config)
# for item in unknown:
#     if ':' in item:
#         mo_name, mo_value = item.split(':')
#         paras.__dict__[mo_name] = int(mo_value)
print("finetune", paras)
run_dir, log_dir = make_logpath('foil', paras.algo)
dir_name=f"[AD]{paras.AD}[St]{paras.St}[theta]{paras.theta}[Phi]{paras.Phi}"
### start env
# env = foil_env(paras, info=f"{args.name}", local_port=8686)
if 'ip' in paras.__dict__:
    env = foil_env(paras, info=f"{dir_name}", network_port=paras['ip'])
else:
    env = foil_env(paras, info=f"{dir_name}", local_port=8686)
paras.action_space, action_dim = env.action_dim, env.action_dim
paras.obs_space, observation_dim = env.observation_dim, env.observation_dim

env.reset()

# agent = SAC(paras)
agent = SinPolicy(paras)
ret = []
obs = env.reset()
Gt = 0
done = False
epsoide_length = 0
epsoide_num = 0
time_cost = 0
info= {}
info['dt'], info['delta_y'], info['delta_theta'] = 0.02, 0, 0
for i in tqdm(range(paras.epochs)):
    # rollout in env  -  rollout()
    ct_sum, eta_sum, cp_sum, fx_sum, dt_sum = 0, 0, 0, 0, 0
    ct_ls, cp_ls, fx_ls, dt_ls = [], [], [], []
    while not done:
        action = agent.choose_action(time_cost, dt=info['dt'], info=info)
        print("action", action, "delta_y", info['delta_y'], "delta_theta", info['delta_theta'])
        
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        Gt += reward
        ct_sum += info['ct']
        eta_sum += info['eta']
        cp_sum += info['cp']
        fx_sum += info['fx']
        dt_sum += info['dt']
        time_cost += info['dt']
        fx_ls.append(info['fx'])
        cp_ls.append(info['cp'])
        ct_ls.append(info['ct'])
        dt_ls.append(info['dt'])
        epsoide_length += 1
        if epsoide_length % 1000 ==0:
            print(f"step:{i}")
        if epsoide_length % 10 ==0:
            pass
            #wandb.log({"reward": reward, "ct_step": info['ct'], "eta_step": info['eta']})
        #print(f"{epsoide_length*0.005}: [action]{action} [ct]{info['ct']} [cp]{info['cp']} [eta]{info['eta']} [obs]{obs}")
    obs = env.reset()
    done = False
    ret.append(Gt)
    avg_reward = (Gt - reward) / epsoide_length
    eff_avg_reward = avg_reward if epsoide_length > 8000 / paras.action_interval else -8000+epsoide_length  # set too short episode to zero
    # cal ct, eta, cp, fx from the middle
    ct_avg = np.average(ct_ls[int(epsoide_length/2):], weights=dt_ls[int(epsoide_length/2):])
    eta_avg = -1* np.average(fx_ls[int(epsoide_length/2):], weights=dt_ls[int(epsoide_length/2):]) / np.average(cp_ls[int(epsoide_length/2):], weights=dt_ls[int(epsoide_length/2):])
    cp_avg = np.mean(cp_ls)
    fx_avg = np.mean(fx_ls)
    #wandb.log({"Gt": Gt, "length": epsoide_length, "episode_num": epsoide_num, "avg_reward": avg_reward,"effective_avg": eff_avg_reward, "ct": ct_sum / epsoide_length,'eta': fx_sum/cp_sum})  # avg should remove the final step reward, since it can be too large
    print("epoch:", i, "length:", epsoide_length, "G:", Gt)
    print("action:", action, "; state: ", next_obs, "; reward:", reward)
    print({"Gt": Gt, "length": epsoide_length, "episode_num": epsoide_num, "avg_reward": avg_reward,
               "effective_avg": eff_avg_reward, "ct": ct_avg,
               'eta': eta_avg, "cp": cp_avg, "fx": fx_avg})
    # write to save csv file  st ad theta phi ct cp eta length  Gt avg_reward
    with open(r'./search.csv', mode='a', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        wf.writerow([paras.St, paras.AD, paras.theta, paras.Phi, ct_avg, cp_avg, eta_avg, epsoide_length, Gt, avg_reward, cp_avg, fx_avg, np.sum(dt_ls)])
    Gt = 0
    epsoide_length = 0
    epsoide_num += 1
    # update policy   -  update()
    #agent.learn()
    # eval policy performance - rollout(train=False)
    # logger
    # if i % 10 == 0:

    # save model parameter
    #if i % 1000 == 0:
    #    agent.save(run_dir, i)
env.close()

#  ct = f.x /contatnt
# eta = fx/ cp p
# ct , -cp
