from framework.utils import set_seed, ConfigDict, make_logpath
from framework.trainer1 import RSAC
from framework import utils
from datetime import datetime, timedelta
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
import gym
import pickle


### load config AND prepare logger
dir = "config/rsac_debug1.yaml"
config_dict = utils.load_config(dir)
paras = utils.get_paras_from_dict(config_dict)
# env = gym.vector.make('foil-v0', config=paras,num_envs=3)
# print("local:", paras)
# wandb.init(project="foil", config=paras, name=args.name)
# paras = utils.get_paras_from_dict(wandb.config)
print("finetune", paras)
run_dir, log_dir = make_logpath('foil', paras.algo)
writer = SummaryWriter(run_dir)
### start env
env = gym.make('Pendulum-v0')#foil_env(paras)
# env = gym.vector.make('foil-v0', num_envs=3)
paras.action_space, action_dim = env.action_space.shape[0], env.action_space.shape[0]
paras.obs_space, observation_dim = env.observation_space.shape[0], env.observation_space.shape[0]

env.reset()

agent = RSAC(paras)
obs = env.reset()
Gt = 0
done = False
epsoide_length = 0
epsoide_num = 0
step_count = 0

buffer_new_data = 0
for i in tqdm(range(15000)):
    agent.reset_rnn()
    while not done:
        action = agent.choose_action(obs)
        next_obs, reward, done, info = env.step(action["action"][0])
        agent.add_experience(
            {"states": np.array(obs).astype(float).reshape(1, observation_dim),
             "states_next": np.array(next_obs).astype(float).reshape(1, observation_dim), "rewards": reward,
             "dones": np.float32(done),
             "id": epsoide_length})
        obs = next_obs
        Gt += reward
        epsoide_length += 1
        buffer_new_data += 1
        step_count += 1
        if buffer_new_data > paras.batch_size:
            # it is better to use data for only 1-2 times
            agent.learn()
            buffer_new_data = 0
            # writer.add_scalar("reward", reward, step_count)
            writer.add_scalar("critic_loss", agent.c_loss, step_count)
            writer.add_scalar("actor_loss", agent.a_loss, step_count)
            # writer.add_scalar("alpha_loss", agent.alpha_loss, step_count)

    obs = env.reset()
    done = False
    avg_reward = (Gt - reward) / epsoide_length
    eff_avg_reward = avg_reward if epsoide_length > 8000 / paras.action_interval else 0  # set too short episode to zero
    # wandb.log({"Gt": Gt, "length": epsoide_length, "episode_num": epsoide_num, "avg_reward": avg_reward,
    #            "effective_avg": eff_avg_reward,
    #            "value": agent.value(obs),
    #            "policy_entropy": agent.ent, "alpha": agent.alpha.detach().cpu().numpy()})  # avg should remove the final step reward, since it can be too large
    # writer.add_scalars("report", tag_scalar_dict={
    #     "avg_reward": avg_reward,
    #     "Gt": Gt,
    #     "length": epsoide_length,
    #     "value": agent.value(obs),
    #     "policy_entropy": agent.ent,
    #     "alpha": agent.alpha.detach().cpu().numpy()
    # },
    # global_step=i)
    # writer.add_scalar("avg_reward", avg_reward, step_count)
    writer.add_scalar("Gt", Gt, step_count)
    # writer.add_scalar("value", agent.value(obs), step_count)
    # writer.add_scalar("policy_entrop", agent.ent, step_count)
    # writer.add_scalar("alpha", agent.alpha.detach().cpu().numpy())
    print("epoch:", i, "length:", epsoide_length, "G:", Gt, "actor_loss:", agent.a_loss, "critic_loss", agent.c_loss)
    print("action:", action, "; state: ", next_obs, "; reward:", reward)
    Gt = 0
    epsoide_length = 0
    epsoide_num += 1
    # update policy   -  update()
    # agent.learn()
    # eval policy performance - rollout(train=False)
    # logger
    # if i % 10 == 0:

env.close()
