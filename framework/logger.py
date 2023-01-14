import logging
import torch.distributed as dist
import wandb
import time
import torch
import numpy as np
from tensorboardX.writer import SummaryWriter
import os
import random

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class Logger():
    def __init__(self, cuda=False):
        self.logger = logging.getLogger(__name__)
        self.cuda = cuda

    def info(self, message, *args, **kwargs):
        if (self.cuda and dist.get_rank() == 0) or not self.cuda:
            self.logger.info(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)


class LogClient(object):
    """
    A logger wrapper with buffer for visualized logger backends, such as tb or wandb
    counting
        all None valued keys are counters
        this feature is helpful when logging from model interior
        since the model should be step-agnostic
    Sets seed for each process
    Centralized saving
    economic logging
        stores the values, log once per log_period
    syntactic sugar
        supports both .log(data={key: value}) and .log(key=value)
    multiple backends
        forwards logging to both tensorboard and wandb
    logger hiearchy and multiagent multiprocess logger
        the prefix does not end with "/"
        prefix = "" is the root logger
        prefix = "*/agent0" ,... are the agent loggers
        children get n_interaction from the root logger
    """

    def __init__(self, server, prefix=""):
        self.buffer = {}
        if isinstance(server, LogClient):
            prefix = f"{server.prefix}/{prefix}"
            server = server.server
        self.server = server
        self.prefix = prefix
        self.log_period = server.getArgs().log_period
        self.last_log = 0
        setSeed(server.getArgs().seed)

    def child(self, prefix=""):
        return LogClient(self, prefix)

    def flush(self):
        self.server.flush(self)
        self.last_log = time.time()

    def log(self, raw_data=None, **kwargs):
        if raw_data is None:
            raw_data = {}
        raw_data.update(kwargs)

        data = {}
        for key in raw_data:  # also logs the mean for histograms
            data[key] = raw_data[key]
            if isinstance(data[key], torch.Tensor) and len(data[key].shape) > 0 or \
                    isinstance(data[key], np.ndarray) and len(data[key].shape) > 0:
                data[key + '_mean'] = data[key].mean()

        # updates the buffer
        for key in data:
            if data[key] is None:
                if not key in self.buffer:
                    self.buffer[key] = 0
                self.buffer[key] += 1
            else:
                valid = True
                # check nans
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].detach().cpu()
                    if torch.isnan(data[key]).any():
                        valid = False
                elif np.isnan(data[key]).any():
                    valid = False
                if not valid:
                    print(f'{key} is nan!')
                    # pdb.set_trace()
                    continue
                self.buffer[key] = data[key]

        # uploading
        if self.server.online or (time.time() > self.log_period + self.last_log):
            self.flush()

    def raw_log(self, **kwargs):
        self.server.logger(**kwargs)

    def save(self, model, info=None):
        state_dict = model.state_dict()
        state_dict = {k: state_dict[k].cpu() for k in state_dict}
        self.server.save({self.prefix: state_dict}, info)

    def getArgs(self):
        return self.server.getArgs()


class LogServer(object):
    """
    We do not assume the logging backend (e.g. tb, wandb) supports multiprocess logging,
    therefore we implement a centralized log manager

    It should not be directly invoked, since we want to hide the implementation detail (.log.remote)
    Wrap it with prefix="" to get the root logger

    It also keeps track of the global step
    """

    def __init__(self, args, mute=False, online=True):
        self.group = args.config['name']
        self.name = args.name
        self.online = online
        if not mute:
            run = wandb.init(
                project="foil",
                config=Config(args)._toDict(recursive=True),
                name=self.name,
                group=self.group,
            )
            self.logger = run
            self.writer = SummaryWriter(log_dir=f"runs/{self.name}")
            self.writer.add_text("config", f"{args}")

        self.mute = mute
        self.args = args
        self.args.log_period = args.config['save_period']
        self.save_period = self.args.log_period
        self.last_save = time.time()
        self.state_dict = {}
        self.step = 0
        self.step_key = 'interaction'
        exists_or_mkdir(f"checkpoints/{self.name}")

    def getArgs(self):
        return self.args

    def flush(self, logger=None):
        if self.mute:
            return None
        if logger is None:
            logger = self
        buffer = logger.buffer
        data = {}
        for key in buffer:
            if key == self.step_key:
                self.step = buffer[key]
            log_key = logger.prefix + "/" + key
            while log_key[0] == '/':
                # removes the first slash, to be wandb compatible
                log_key = log_key[1:]
            data[log_key] = buffer[key]

            if isinstance(data[log_key], torch.Tensor) and len(data[log_key].shape) > 0 or \
                    isinstance(data[log_key], np.ndarray) and len(data[log_key].shape) > 0:
                self.writer.add_histogram(log_key, data[log_key], self.step)
            else:
                self.writer.add_scalar(log_key, data[log_key], self.step)
            self.writer.flush()

        self.logger.log(data=data, step=self.step, commit=False)
        # "warning: step must only increase "commit = True
        # because wandb assumes step must increase per commit
        self.last_log = time.time()

    def save(self, state_dict=None, info=None, flush=True):
        if not state_dict is None:
            self.state_dict.update(**state_dict)
        if flush and time.time() - self.last_save >= self.save_period:
            filename = f"{self.step}_{info}.pt"
            if not self.mute:
                with open(f"checkpoints/{self.name}/{filename}", 'wb') as f:
                    torch.save(self.state_dict, f)
                print(f"checkpoint saved as {filename}")
            else:
                print("not saving checkpoints because the logger is muted")
            self.last_save = time.time()

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Config(object):
    def __init__(self, args):
        self.args = args
        return None


    def _toDict(self, recursive=False):
        """
            converts to dict for **kwargs
            recursive for logging
        """
        pr = {}
        for name in dir(self.args):
            value = getattr(self.args, name)
            if not name.startswith('_') and not name.endswith('_'):
                #if isinstance(value, dict) and recursive:
                #    value = value._toDict(recursive)
                pr[name] = value
        return pr