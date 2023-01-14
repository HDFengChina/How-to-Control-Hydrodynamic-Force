import pandas as pd
import numpy as np
import pickle
from model.sin_policy import SinPolicy
from env.flow_field_env import fake_env, foil_env
from framework import utils

mode = "env" # "data" or "env"

if mode == "data":
# Load the data from the csv file
    episode_name = '[AD]0.9[St]0.1[theta]50[Phi]90'
    path = f"/mnt/nasdata/runji/LilyPad/RLnonparametric_Foil1/saved_backup/{episode_name}/Episode3/output.csv"
    df_csv = pd.read_csv(path)

    # Load the data from the txt file
    path = "/mnt/nasdata/runji/LilyPad/RLnonparametric_Foil1/saved_backup/[AD]0.9[St]0.1[theta]50[Phi]90/Episode3/force.txt"
    df_txt = pd.read_csv(path)

    # remove the first three line in dataframe
    df_txt = df_txt[3:]
    # spilt each line into a list
    df_txt = df_txt.apply(lambda x: x.str.split(' ').tolist())
    # remove the second element in the list in the first row
    temp_title = list(df_txt.iloc[0, :])[0]
    temp_title.remove('')
    df_txt.iloc[0, :] = [temp_title]

    df_txt = pd.DataFrame(df_txt.iloc[1:, 0].tolist(), columns=temp_title)
    df = pd.concat([df_csv, df_txt], axis=1)
    # spilt 'SP' in the column 'SparsePressure' into two columns
    df['SP'] = df['SP'].str.split('_')
    sp = np.array(df['SP'])
    sp_ls= []
    for item in sp:
        temp = []
        for i in item:
            i = i if ';' not in i else i[:-1]
            temp.append(float(i))
        sp_ls.append(temp)
    # convert string into float in the dataframe except the last column
    df[df.columns[:-1]] = df[df.columns[:-1]].apply(pd.to_numeric)
    action = np.array(df[['Action1', 'Action2']])
    state = {'y': np.array(df['y']), 'theta': np.array(df['theta']), 'yvel': np.array(df['yvel']), 'tvel': np.array(df['tvel']), 'SP': sp_ls}

    # save the data into a pickle file
    with open(f'D:\dataset\Lilypad\{episode_name}.pkl', 'wb') as f:
        pickle.dump([action, state], f)
    print(f'{episode_name} is saved')
else:
    dir = "config/sin.yaml"
    config_dict = utils.load_config(dir)
    paras = utils.get_paras_from_dict(config_dict)
    print("local:", paras)
    episode_name = f"[AD]{paras.AD}[St]{paras.St}[theta]{paras.theta}[Phi]{paras.Phi}"
    # interacte with environment
    env = foil_env(paras, info=f"{episode_name}", local_port=8686)
    agent = SinPolicy(paras)
    obs = env.reset()
    epsoide_length = 0
    action_ls, state_ls = [], []
    done= False
    time_cost = 0
    info= {}
    info['dt'] = 0
    while not done:
        action = agent.choose_action(time_cost, dt=info['dt'])#epsoide_length*0.005) # 0.005 is the time step but in the dynamic mode you should change the time step
        next_obs, reward, done, info = env.step(action)
        action_ls.append(action)
        state_ls.append(obs)
        time_cost += info['dt']
        obs = next_obs
        epsoide_length += 1
        print(f'epsoide_length: {epsoide_length}, time_cost: {time_cost}, reward: {reward}, action: {action}, state: {obs}')
    # save the data into a pickle file
    with open(f'D:\dataset\Lilypad\{episode_name}_online.pkl', 'wb') as f:
        pickle.dump([action_ls, state_ls], f)
