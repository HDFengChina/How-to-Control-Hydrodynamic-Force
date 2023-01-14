from xmlrpc.client import ServerProxy
import subprocess
import json
import time
import random
import numpy as np
import argparse
from gym.spaces import Box
import os
import signal


def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


class foil_env:
    def __init__(self, config=None, info='', local_port=None, network_port=None):
        self.observation_dim = 2
        self.action_dim = 3
        self.step_counter = 0
        self.state = [0, 0]
        self.action_interval = config.action_interval  # 1
        # for gym vector env
        self.unwrapped=self
        self.unwrapped.spec = None
        self.observation_space = Box(low=-1e6,high=1e6,shape=[self.observation_dim])
        self.action_space = Box(low=-1,high=1,shape=[self.action_dim])
        self.local_port = local_port

        # TODO:start the server
        while True:
            port = random.randint(6000, 8000)
            if not is_port_in_use(port):
                break
        port = port if network_port == None else network_port
        # print("Foil_env start!")
        if local_port == None:
            self.server = subprocess.Popen(f'xvfb-run -a /home/msrai4scipde/home/processing-3.5.4/processing-java --sketch=/home/msrai4scipde/home/short/PinballCFD_server --run {port} {info}', shell=True)
            # wait the server to start
            time.sleep(20)
            print("server start")
            self.proxy = ServerProxy(f"http://localhost:{port}/")
        else:
            self.proxy = ServerProxy(f"http://localhost:{local_port}/")

    def step(self, action):
        self.step_counter += 1
        result_ls, ct_ls, eta_ls, cp_ls, fx_ls, dt_ls = [], [], [], [], [], []
        for i in range(self.action_interval):  # only 1
            step_1 = float(action[0])  # /self.action_interval # since velosity # 前柱速度
            step_2 = float(action[1])  # /self.action_interval # 下柱速度
            step_3 = float(action[2])  # /self.action_interval # since velosity # 上柱速度
            action_json = {"v1": step_1, "v2": step_2, "v3": step_3}
            res_str = self.proxy.connect.Step(json.dumps(action_json))  # 将action传入lilypad并返回state, reward, 将一个Python数据结构转换为JSON
            [state, reward, done] = self.parseStep(res_str)
            self.reward, self.state, self.done = np.array(reward, dtype=np.float32), np.array(state, np.float32), np.array(done, np.float32)
            result_ls.append(self.reward)
            print("done:", self.done)
            #print(result_ls)
            if self.done == True:
                break
        self.reward = np.average(result_ls) ## TODO: mean?
        #print(self.reward)
        #print(res)
        return self.state, self.reward, self.done, self.reward

    def reset(self):
        # TODO: reset true environment
        action_json = {"v1": 0, "v2": 0, "v3": 0}
        res_str = self.proxy.connect.reset(json.dumps(action_json))
        state, reward, done = self.parseStep2(res_str)
        self.reward, self.state, self.done = np.array(reward, dtype=np.float32), np.array(state, np.float32), np.array(done, np.float32)
        return self.state

    def parseStep2(self, info):  # 对lilypad返回信息解码
        all_info = json.loads(info)
        state = json.loads(all_info['state'][0])
        reward = all_info['reward']
        done = all_info['done']
        state_ls = [state['lift'], state['drag']]  # state
        # state_ls = list(np.nan_to_num(np.array(state_ls), nan=0))
        return state_ls, reward, done
        
    def parseStep(self, info):  # 对lilypad返回信息解码
        all_info = json.loads(info)
        # print("####################info", all_info['state'])
        # try:
        # state = json.loads(all_info['state'][0])
        # print("####################state", state)
        lift = all_info['cl']
        drag = all_info['cd']
        reward = all_info['reward']
        done = all_info['done']
        state_ls = [lift, drag]  # state
        # state_ls = list(np.nan_to_num(np.array(state_ls), nan=0))
        # except:
            # print("########out of index happen#############")
        return state_ls, reward, done

    def parseState(self, state):
        state = json.loads(json.loads(state)['state'][0])
        state['SparsePressure'] = list(map(float, state['SparsePressure'].split('_')))

        ## TODO: define different combination for state
        state_ls = [state['delta_y'], state['y_velocity'], state['eta'], state['delta_theta'], state['x_velocity'],
                    state['theta_velocity']] + state['SparsePressure']
        state_ls = list(np.nan_to_num(np.array(state_ls), nan=0))
        return state_ls

    def terminate(self):
        pid = os.getpgid(self.server.pid)
        self.server.terminate()
        os.killpg(pid, signal.SIGTERM)  # Send the signal to all the process groups.

    def close(self):
        if self.local_port == None:
            self.server.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    args.action_interval = 1
    env = foil_env(args)
    env.reset()
    done = False
    while done != True:
        [state, reward, done] = env.step([0.27, 0.97, 0.11])
        print(done)

    env.step([0, 0, 0])
    env.reset()
