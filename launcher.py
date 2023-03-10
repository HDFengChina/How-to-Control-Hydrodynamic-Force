import os
import time

mode = 'finetune'  # 'final' or 'finetune'

env_ls = [ 'slowdown' ] #'catchup', 'eight'
algo_ls = ['IC3Net']
name = '[finetune-4]'

AD = [5]
St = [0.1]
theta = [70]
Phi = []
ip = [8686]


#command = f'python run_foil_sequence AD:{} St:{} theta:{} Phi:{} ip:{} '
session_name = "foil_2"
command = f'tmux new-session -d -t {session_name}'
os.system(command)
os.system('tmux send-keys clear')
os.system('tmux send-keys KPEnter')
for i in range(50):
    # set a new window to contain a specific experiment
    os.system(f'tmux new-window -d -n actor_{i}')

    # prepare the python running command
    py_command = f"tmux  send-keys -t actor_{i} 'wandb agent linrunji/foil_search/2869t010' ENTER"
    os.system(py_command)
    time.sleep(10)


os.system(f"tmux attach -t {session_name}")
"""
if mode == 'final':
    seed_number = 5
    for algo in algo_ls:
        for env in env_ls:
            # set a new tmux session to contain a algo-env group
            command = f'tmux new-session -d -t {algo}_{env}{name}'
            os.system(command)
            os.system('tmux send-keys clear')
            os.system('tmux send-keys KPEnter')
            for i in range(seed_number):
                # set a new window to contain a specific experiment
                os.system(f'tmux new-window -d -n actor_{i}')

                # prepare the python running command
                py_command = f"tmux  send-keys -t actor_{i} 'python launcher_baseline.py --env {env} --algo {algo} --name {name}' ENTER"
                os.system(py_command)
elif mode == 'finetune':
    seed_number = 5*8
    tune_ls = {'ring': ['\'{\\\"agent_args.lr_v\\\":3e-3}\'',
                        '\'{\\\"agent_args.lr\\\":2e-3,\\\"agent_args.lr_v\\\":2e-3}\'',
                        '\'{\\\"agent_args.lr\\\":5e-3,\\\"agent_args.lr_v\\\":6e-3}\''],
               'eight': ['\'{\\\"agent_args.lr_v\\\":1e-2,\\\"agent_args.lr_p\\\":1e-2}\'',
                         '\'{\\\"agent_args.lr_v\\\":1e-2,\\\"agent_args.lr_p\\\":1e-3}\'',
                         '\'{\\\"agent_args.lr_v\\\":1e-3,\\\"agent_args.lr_p\\\":1e-3}\''],
               'catchup': ['{}',
                           '\'{\\\"agent_args.lr\\\":5e-4,\\\"agent_args.lr_v\\\":5e-4}\'',
                           '\'{\\\"agent_args.lr\\\":5e-3,\\\"agent_args.lr_v\\\":5e-3}\'',
                           '\'{\\\"agent_args.lr\\\":5e-5,\\\"agent_args.lr_v\\\":5e-4}\''],
               'slowdown': ['\'{\\\"agent_args.lr_v\\\":1e-2,\\\"agent_args.lr_p\\\":1e-2}\'',
                         '\'{\\\"agent_args.lr_v\\\":1e-2,\\\"agent_args.lr_p\\\":1e-3}\'',
                         '\'{\\\"agent_args.lr_v\\\":1e-3,\\\"agent_args.lr_p\\\":1e-3}\''],

               }
    for algo in algo_ls:
        for env in env_ls:
            # set a new tmux session to contain a algo-env group
            command = f'tmux new-session -d -t {algo}_{env}{name}'
            os.system(command)
            os.system('tmux send-keys clear')
            os.system('tmux send-keys KPEnter')
            for i in range(seed_number):
                for j in range(len(tune_ls[env])):
                    # set a new window to contain a specific experiment
                    os.system(f'tmux new-window -d -n actor_para{j}_{i}')

                    # prepare the python running command
                    py_command = f"tmux  send-keys -t actor_para{j}_{i} \"python ./env/flow_field_env.py  \" ENTER"
                    os.system(py_command)
# os.system(f"tmux attach -t {algo}_{env}{name}")
"""
'''
## CPPO
tune_ls = {'ring': ['{\"agent_args.lr_v\":2e-3}',
                        '{\"agent_args.lr\":1e-3}',
                        '{\"agent_args.lr\":2e-3, \"agent_args.lr_v\":2e-3}'],
               'eight': ['{}'],
               'catchup': ['{}'],
               'slowdown': ['{}',
                        '{\"agent_args.lr\":1e-4,"agent_args.lr_v":1e-3}'],
               }
tune_ls = {'ring': ['\'{\\\"agent_args.lr_v\\\":3e-3}\'',
                        '\'{\\\"agent_args.lr\\\":1e-3',
                        '\'{\\\"agent_args.lr\\\":5e-3,\\\"agent_args.lr_v\\\":6e-3}\''],
               'eight': ['{}'],
               'catchup': ['{}'],
               'slowdown': ['{}',
                        '\'{\\\"agent_args.lr\\\":1e-4,\\\"agent_args.lr_v\\\":1e-3}\''],

               }
# for eight               

## IA2C
tune_ls = {'ring': ['\'{\\\"agent_args.lr_v\\\":3e-3}\'',
                        '\'{\\\"agent_args.lr\\\":2e-3,\\\"agent_args.lr_v\\\":2e-3}\'',
                        '\'{\\\"agent_args.lr\\\":5e-3,\\\"agent_args.lr_v\\\":6e-3}\''],
               'eight': ['{}'],
               'catchup': ['{}',
                           '\'{\\\"agent_args.lr\\\":5e-4,\\\"agent_args.lr_v\\\":5e-4}\'',
                           '\'{\\\"agent_args.lr\\\":5e-3,\\\"agent_args.lr_v\\\":5e-3}\'',
                           '\'{\\\"agent_args.lr\\\":5e-5,\\\"agent_args.lr_v\\\":5e-4}\''],
               'slowdown': ['{}',
                        '\'{\\\"agent_args.lr\\\":1e-4,\\\"agent_args.lr_v\\\":1e-3}\''],

               }
'''