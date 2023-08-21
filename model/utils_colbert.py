import json
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('utils')


def get_model_name(args):
    project_name = args.data_dpath.split('/')[-1]
    training_set = args.triples.split('/')[-1].split('_')[-2]
    if 'bert-base-uncased' in args.config:
        config = 'bertbase'
    elif 'BERTOverflow' in args.config:
        config = 'bertoverflow'
    elif 'codebert' in args.config:
        config = 'codebert'
    return 'model_SemanticCodebert_' + '_'.join(
        [project_name, training_set, config, args.special_tokens, 'q' + str(args.query_maxlen),
         'd' + str(args.doc_maxlen), 'dim' + str(args.dim), args.similarity, args.granularity])


def create_directory(path):
    if os.path.exists(path):
        logger.info('#> Note: Output directory {0} already exists'.format(path))
    else:
        print('\n')
        logger.info('#> Creating directory {0}'.format(path))
        os.makedirs(path)


def get_special_tokens(config):
    tokens = config.split('/')[-1].split('_')[5]
    assert tokens == 'QARCL' or tokens == 'QARC' or tokens == 'QD' or tokens == 'QARCD'
    return tokens


def get_config(config):
    if config == 'BERT':
        config = 'bert-base-uncased'
        logger.info('Running with BERT NL')
    elif config == 'BERTOverflow':
        config = '../BERTOverflow'
        logger.info('Running with BERTOverflow')
    elif config == 'CodeBERT':
        config = 'microsoft/codebert-base'
        logger.info('Running with CodeBERT')
    else:
        raise ValueError('Unknown config {0}'.format(config))
    logger.info('Fine tune BERT model loaded from {0}'.format(config))
    return config


def load_br2ts(data_dpath, ts_type='open'):
    if ts_type == 'open':
        fname = 'open_ts.txt'
    elif ts_type == 'fix':
        fname = 'fix_ts.txt'
    else:
        raise RuntimeError('Unknown timestamp type: {0}'.format(ts_type))

    bug2ts = dict()
    with open(os.path.join(data_dpath, fname)) as f:
        for line in f.readlines():
            bid, ts = line.strip().split(',')
            bid = int(bid)
            ts = int(float(ts))
            bug2ts[bid] = ts
    return bug2ts


def load_commit2ts(data_dpath):
    commits2ts = dict()
    for fname in os.listdir(os.path.join(data_dpath, 'commits')):
        if not fname.startswith('c_'):
            continue

        with open(os.path.join(data_dpath, 'commits', fname)) as f:
            commit = json.load(f)

        ts = int(commit['timestamp'])
        sha = commit['sha']

        commits2ts[sha] = ts
    return commits2ts
