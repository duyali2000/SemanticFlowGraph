import datetime
import logging
import os
import random
from data_utils import load_brs
import numpy as np
import torch

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('utils')


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def set_seed(seed=1234, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


def get_model_name(args):
    project_name = args.project_path.split('/')[-1]
    model_type = args.model_type
    trainig_set_name = args.training_set.split('_')[-1][:-4]
    special_tokens = args.special_tokens
    if len(special_tokens) == 0:
        special_tokens = 'None'
    return 'model_' + project_name + '_' + model_type + '_' + trainig_set_name + '_' + special_tokens


def get_device(gpu):
    if torch.cuda.is_available() and gpu > -1:
        device = torch.device("cuda:" + str(gpu))
        logger.info("Running on the GPU")
    else:
        device = torch.device("cpu")
        logger.info("Running on the CPU")
    return device


def get_br2skip(project_dpath, evaluate_on='test'):
    if evaluate_on != 'all':
        bug_ids, texts, br2ts = load_brs(project_dpath)
        br_ts = [(k, v) for k, v in br2ts.items()]
        br_ts = sorted(br_ts, key=lambda x: x[1])
        if evaluate_on == 'train':
            # first half of BRs is always used for training, so we need to skip second half
            br_to_skip = set([x[0] for x in br_ts[int(len(br_ts) / 2):]])
        elif evaluate_on == 'test':
            # second half of BRs creates a testing set, so we need to skip first half
            br_to_skip = set([x[0] for x in br_ts[:int(len(br_ts) / 2)]])
        else:
            raise ValueError('Evalution options are: all, train, test.')
    else:
        br_to_skip = None

    return br_to_skip


def log_time(project_dpath, task, config_name, time_delta):
    time_dpath = os.path.join(project_dpath, 'timer', task)
    if not os.path.exists(time_dpath):
        os.makedirs(time_dpath)

    with open(os.path.join(time_dpath, 'time_{0}'.format(config_name)), 'w') as f:
        f.write(str(time_delta.total_seconds()))