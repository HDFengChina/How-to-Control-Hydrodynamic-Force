import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pickle
from xmlrpc.server import SimpleXMLRPCServer
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from agent_TD3 import Agent
from utils import KalmanFilter, NoneFilter
from datetime import datetime
import argparse
import itertools


class Server():
    def __init__(self, environment, filter, explore_noise, one_action, *args, **kwargs):
        self.train_t = 0
        self.environment = environment
        self.one_action = one_action
        self.time_step = 0
        self.collect_n = 64
        self.state_dim = 64
        if self.one_action:
            self.action_dim = 1
        else:
            # self.action_dim = 2
            self.action_dim = 3
        self.agent = Agent(self.state_dim, self.action_dim, explore_noise = explore_noise)
        self._stamp("Environment: " + environment)
        self._stamp("Using Exploration Noise: " + explore_noise)
        self.state_record = list(np.ones((self.collect_n,1))[:,np.newaxis])  # filtered.
        self.state_combine = []
        self.reward_record = []
        self.unfiltered_state_record = []
        self.action_record = []
        self.state_mean = np.zeros((1,self.state_dim))
        # self.server = SimpleXMLRPCServer(("0.0.0.0", 8000))
        # self.server = SimpleXMLRPCServer(("localhost", 8000),logRequests=False)
        # self._register()
        self.filter_name = filter
        if self.filter_name == "Kalman":
            self.filter = KalmanFilter()
            self._stamp("Using Kalman Filter")
        elif self.filter_name == "None":
            self.filter = NoneFilter()
            self._stamp("Using No Filter")
        else:
            raise NotImplementedError
        self.save_model_dir = "save_127"
        self.save_data_dir = "save_data_127"
        self.save_eval_dir = "save_eval_3"


    def _stamp(self, string):
        time = "UTC " + datetime.utcnow().isoformat(sep=" ", timespec="milliseconds") + " "
        print(time + string, flush = True)

    def _register(self):
        self.server.register_function(self._init, "init")
        self.server.register_function(self._start_episode, "start_episode")
        self.server.register_function(self._request_stochastic_action, "request_stochastic_action")
        self.server.register_function(self._request_deterministic_action, "request_deterministic_action")
        self.server.register_function(self._train, "train")
        self.server.register_function(self._save, "save")
        self.server.register_function(self._save_eval, "save_eval")
        self.server.register_function(self._restore, "restore")

    def _reward_func(self, this_state, action, next_state):
        # reward = -np.sign(next_state[0,31]) * (next_state[0,31])**2
        # reward = (np.sign(next_state[0,62]) * pow(abs(next_state[0,62]),0.5))
        # reward = np.sign(next_state[0,62]) * (next_state[0,62])**2
        # reward = -next_state[0,62]
        # reward = -np.sign(next_state[0,63]) * pow(abs(next_state[0,63]),0.5) - 0.1*(next_state[0,62])**2
        # reward = -np.sign(next_state[0,63]) * pow(abs(next_state[0,63]),0.5) # save 15的reward
        reward = -np.sign(next_state[0,63]) * pow(abs(next_state[0,63]),0.5) + np.sign(next_state[0,62]) * pow(abs(next_state[0,62]),0.5) # save 15的reward with front, 17 with top
        # print("######################cd###########",(-np.sign(next_state[0,63]) * pow(abs(next_state[0,63]),0.5)))
        # print("######################cl###########",(-0.1*(next_state[0,62]-2)**2))
        # reward = -(next_state[0, 62])**2
        # print("########reward#########", reward)
        # reward = np.sign(next_state[0,31]) * next_state[0,31]**2
        # reward = -(np.sign(next_state[0, 1]) * (next_state[0, 1]) ** 2)
        # print(next_state)
        # print(next_state[0, 31])
        # reward = -(next_state[0, 63]-1) ** 2  #first reward tried
        # reward = -(next_state[0, 31] - 5) ** 2 + 1   #second reward
        # print(reward)
        # reward = -(next_state[0, 31] - 5) ** 2 - abs(next_state[0, 31]-next_state[0, 29]) #third reward tried
        # reward = -(next_state[0, 31] - 5) ** 2 - abs(next_state[0, 31] - next_state[0, 29]) + 1.5  # forth reward
        # reward = -(next_state[0, 31] - 5) ** 2 - (abs(next_state[0,:-1] - this_state[0,:-1])).mean()  # fifth reward
        # reward = -4*(next_state[0, 31] - 5) ** 2 - (next_state[0, 31] - 4.5) ** 2 - (next_state[0, 31] - 5.5) ** 2  # sixth reward
        # print((abs(next_state[0,:-1] - this_state[0,:-1])).mean())
        # print(reward)
        # reward = -np.sign(next_state[0,1]) * (next_state[0,1])**2 - 0.1*next_state[0,0]**2

        # reward_raw = - next_state[0,1] - np.pi * (1/8) * 0.0097 * (3.66**3) * np.sum((np.abs(action)**3))
        return reward

    def _converter(self, signal):
        """
        convert the signal from sensor to forces and moment.
        Args: 
            signal list: 6*n, 
        Return:
            states (ndarray): (n,2), dimensionless C_lift and C_drag 
        """
        non_dimensional = 0.5 * 1000 * 0.2**2 * 2 * 20 * 0.0254**2
        calmat = np.array([
        [-0.03424, 0.00436,  0.04720,  -1.87392,  0.00438,  1.89084],
        [-0.02353,  2.19709,  -0.02607,  -1.08484,  0.03776,  -1.09707],
        [3.32347,  -0.13564,  3.36623,  -0.09593,  3.35105,  -0.15441],
        [-0.00778,  1.04073,  -3.83509,  -0.40776,  3.81653,  -0.69079],
        [4.36009,  -0.17486,  -2.26568,  0.94542,  -2.18742,  -0.79758],
        [0.03483,  -2.32692,  0.02473,  -2.30260,  0.01156,  -2.33458]])

        signal_array = np.mean(np.array(signal).reshape((6,-1)), axis = 1, keepdims = True)
        calforce = calmat @ signal_array
        lift = calforce[0:1,:]*4.44822 / non_dimensional # C_l
        drag = calforce[1:2,:]*4.44822 / non_dimensional # C_d
        state = np.concatenate([lift, drag], axis = 0) #(state_dim, 1)

        return state.transpose() #(1, state_dim)

    def _init(self, episode_count):
        try:
            self._restore(episode_count)
            return True

        except Exception:
            self.agent.reset_agent()
            self._stamp("Initialized!")
            return False

    def _start_episode(self,raw_data):
        # raw_data for calibrating
        if self.environment == "Experiment":
            self.state_mean = self._converter(raw_data) #(1, state_dim)
        self.state_record = list((np.ones((self.collect_n,1))*1)[:,np.newaxis])
        self.unfiltered_state_record = []
        self.action_record = []
        self.state_combine = []
        self.time_step = 0
        self.agent.reset_episode()
        self.filter.reset()
        self._stamp("Epsode Start!")
        return True

    def _request_action(self, raw_data, stochastic):
        # if self.environment == "Experiment":
        #     calibrated_state = self._converter(raw_data) - self.state_mean #(1, state_dim)
        # elif self.environment == "CFD":
        #     calibrated_state = np.asfarray(raw_data.split("_"),float)[None,:]
        # else:
        #     raise NotImplementedError
        filtered_state = self.filter.estimate(raw_data)
        

        # print(filtered_state)

        self.unfiltered_state_record.append(raw_data)
        self.state_record.append(filtered_state[0][0])  # lift
        self.state_record.append(filtered_state[0][1])  # drag

        # print(self.state_record)

        combined_state = list(np.array(self.state_record[(self.time_step+1)*2:(self.time_step+1)*2+64], dtype=object).flatten())

        # print(self.time_step)
        # print(combined_state)
        # combined_state.append(self.time_step)
        # print(combined_state)
        self.time_step += 1
        # print(np.array(combined_state)[np.newaxis,:])
        self.state_combine.append(np.array(combined_state, dtype=object)[np.newaxis,:])
        # print(self.state_combine)

        action = self.agent.get_action(np.array(combined_state, dtype=object)[np.newaxis,:], stochastic=stochastic) 

        if self.one_action:
            action = np.concatenate((action, - action), axis=1)

        self.action_record.append(action)
        self._stamp("state: " + np.array2string(filtered_state[0,:], formatter={'float_kind':lambda x: "%.2f" % x}) +
                    "  action: " + np.array2string(action[0,:], formatter={'float_kind':lambda x: "%.2f" % x}))

        #if self.environment == "Experiment":
        #    raw_action = action[0,:].tolist()
        #elif self.environment == "CFD":
        #    raw_action = '_'.join([str(i) for i in action[0,:]])
        #else:
        #    raise NotImplementedError
        #print(type(action))

        return action

    def _request_stochastic_action(self, raw_data):
        return self._request_action(raw_data, True)

    def _request_deterministic_action(self, raw_data):
        return self._request_action(raw_data, False)
       
    def _train(self, steps): #step=1000
        reward_sum = []
        if not os.path.exists(self.save_data_dir):
            os.mkdir(self.save_data_dir)
        np.savez(self.save_data_dir + "/data_{}.npz".format(self.agent.episode_count),
                    state = np.array(self.state_record),
                    unfiltered_state = np.array(self.unfiltered_state_record),
                    action = np.array(self.action_record))
        print(self.agent.episode_count)
                    
        record_length = len(self.state_combine)
        self._stamp("Length of Record: " + str(record_length))

        # print(record_length)
        # print(self.state_combine[10])

        for i in range(0, record_length - 1):
            # print(self.state_combine[i])
            reward = self._reward_func(self.state_combine[i],
                                       self.action_record[i], 
                                       self.state_combine[i+1])

            # print(self.state_combine[i])
            # print(self.state_combine[i+1])
            reward_sum.append(reward)

            self.agent.replay_buffer.store(self.state_combine[i],
                                        self.action_record[i][0,:self.action_dim],
                                        reward,
                                        self.state_combine[i+1],
                                        0)
            # print(i)
        # print("train")

        np.savez(self.save_data_dir + "/data_reward_{}.npz".format(self.agent.episode_count),
                    reward = np.array(reward_sum))

        self._stamp("Training Start!")        
        for i in range(steps):
            [q1, q2, qloss] = self.agent.train_iter()

        # print(q)
        # print(len(q1))
        # print(type(q1))

        # np.savez(self.save_data_dir + "/data_Q1_{}.npz".format(self.agent.episode_count),
        #             Q1 = np.array(q1))
        #
        # np.savez(self.save_data_dir + "/data_Q2_{}.npz".format(self.agent.episode_count),
        #             Q2 = np.array(q2))
        #
        # np.savez(self.save_data_dir + "/data_Qloss_{}.npz".format(self.agent.episode_count),
        #             Qloss = np.array(qloss))

        self._stamp("Training End!")        
        return True

    def _save_eval(self, episode_count, rep_count):
        if not os.path.exists(self.save_eval_dir):
            os.mkdir(self.save_eval_dir)
        np.savez(self.save_eval_dir + "/data_{}_{}.npz".format(episode_count, rep_count), 
            state = np.array(self.state_record), 
            unfiltered_state = np.array(self.unfiltered_state_record), 
            action = np.array(self.action_record))

        return True

    def _save(self, dummy = None):
        if not os.path.exists(self.save_model_dir):
            os.mkdir(self.save_model_dir)
        self.agent.saver.save(self.agent.sess, self.save_model_dir + "/{}.ckpt".format(self.agent.episode_count))
        
        # pickle_out = open(self.save_model_dir + "/{}.pickle".format(self.agent.episode_count),"wb")
        # pickle.dump(self.agent.replay_buffer, pickle_out)
        # pickle_out.close()
        self._stamp("Saved Episode {}!".format(self.agent.episode_count))
        self.time_step = 0
        self.state_combine = []
        return True

    def _restore(self,episode_count):
        self.agent.saver.restore(self.agent.sess, self.save_model_dir + "/{}.ckpt".format(episode_count))
        # pickle_in = open(self.save_model_dir + "/{}.pickle".format(episode_count),"rb")
        # self.agent.replay_buffer = pickle.load(pickle_in)
        self.agent.episode_count = episode_count
        self._stamp("Restored from Episode {}!".format(episode_count))
        return True

    def start_server(self):
        self._stamp("Server Listening...")
        print(self.action_dim)
        self.server.serve_forever()

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="server for experiments and CFD")
#    parser.add_argument("-env", "--environment", choices=["CFD", "Experiment"], default="Experiment")
#    parser.add_argument("-fil", "--filter", choices=["None", "Kalman"], default="Kalman")
#    parser.add_argument("-exn", "--explore_noise", choices=["Gaussian", "Process"], default="Gaussian")
#    parser.add_argument("-one", "--one_action", action='store_true')
#    args = parser.parse_args()

#    myserver = Server(**args.__dict__)
#    myserver.start_server()
