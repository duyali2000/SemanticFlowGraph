import argparse
import logging
import os
import sys
import time
from datetime import timedelta

sys.path.append('.')
import torch
import torch.nn as nn
from transformers import AdamW

from SemanticCodebert import SemanticCodebert, EagerBatcher
from manager import MixedPrecisionManager
from utils_colbert import get_model_name, get_config
from utils import get_device, set_seed, log_time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('main')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--special-tokens', choices=['QD', 'QARC', 'QARCL'], default='QARCL')
    parser.add_argument('--gpu', type=int, default=-1)  # -1 == CPU
    parser.add_argument('--n-epochs', type=int, default=4)

    parser.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'])
    parser.add_argument('--dim', dest='dim', default=128, type=int)
    parser.add_argument('--query_maxlen', dest='query_maxlen', default=256, type=int)
    parser.add_argument('--doc_maxlen', dest='doc_maxlen', default=256, type=int)
    parser.add_argument('--mask-punctuation', dest='mask_punctuation', default=True, action='store_true')

    parser.add_argument('--lr', dest='lr', default=3e-05, type=float)
    parser.add_argument('--maxsteps', dest='maxsteps', default=400000, type=int)
    parser.add_argument('--bsize', dest='bsize', default=8, type=int)
    parser.add_argument('--accum', dest='accumsteps', default=2, type=int)
    parser.add_argument('--amp', dest='amp', default=True, action='store_true')

    parser.add_argument('--triples', dest='triples', default='training_dataset_RN_hunks.csv')
    parser.add_argument('--data-dpath', dest='data_dpath', default='../../../data/zxing')
    parser.add_argument('--config', choices=['BERTOverflow', 'BERT', 'CodeBERT'], default='BERTOverflow')

    return parser.parse_args()


def train(args, device):
    set_seed(1234, deterministic=False)

    args.config = get_config(args.config)
    args.granularity = args.triples.split('_')[-1][:-4]

    reader = EagerBatcher(args)

    colbert = SemanticCodebert(args.config, args.special_tokens, dev=device, query_maxlen=args.query_maxlen,
                      doc_maxlen=args.doc_maxlen, dim=args.dim, similarity_metric=args.similarity,
                      mask_punctuation=args.mask_punctuation)

    colbert = colbert.to(device)
    colbert.train()

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)
    criterion = nn.CrossEntropyLoss()
    labels = torch.zeros(args.bsize, dtype=torch.long, device=device)

    train_loss = list()
    start_batch_idx = 0
    t0 = time.time()

    for epoch in range(args.n_epochs):
        logger.info('Epoch {0}/{1}'.format(epoch + 1, args.n_epochs))
        t0_e = time.time()

        for batch_idx, BatchSteps in zip(range(start_batch_idx, args.maxsteps), reader):
            this_batch_loss = 0.0

            for queries, passages in BatchSteps:
                with amp.context():
                    """x = colbert(queries, passages)

                    scores = x.view(2, -1).permute(1, 0)
                    loss = criterion(scores, labels[:scores.size(0)])"""
                    x, y,  output1, target1, output2, target2 = colbert(queries, passages, args.bsize, tag="train")
                    scoresx = x.view(2, -1).permute(1, 0)
                    scoresx /= 0.07
                    loss1 = criterion(scoresx, labels[:scoresx.size(0)])
                    scoresy = y.view(2, -1).permute(1, 0)
                    scoresy /= 0.07
                    loss2 = criterion(scoresy, labels[:scoresy.size(0)])
                    loss3 = criterion(output1, target1) + criterion(output2, target2)
                    loss = loss1 + loss2 + loss3
                    loss = loss / args.accumsteps

                amp.backward(loss)

                train_loss.append(loss.item())
                this_batch_loss += loss.item()

            amp.step(colbert, optimizer)

            if batch_idx % 10 == 0:
                avg_loss = sum(train_loss) / len(train_loss)

                # num_examples_seen = (batch_idx - start_batch_idx) * args.bsize
                logger.info('Batch {1}\ttrain/avg_loss = {0}'.format(avg_loss, batch_idx))
                logger.info('Batch {1}\ttrain/batch_loss = {0}'.format(this_batch_loss, batch_idx))
                # logger.info('Batch {1}\ttrain/examples = {0}'.format(num_examples_seen, batch_idx))

        logger.info('Epoch done! Training took: {0}'.format(str(timedelta(seconds=time.time() - t0_e))))
        reader._reset_triples()

    training_time = timedelta(seconds=time.time() - t0)
    logger.info('Training done! Training took: {0}'.format(str(training_time)))

    model_name = get_model_name(args)
    model_fpath = os.path.join(args.data_dpath, model_name)
    torch.save(colbert.state_dict(), model_fpath)
    logger.info('Model saved to {0}'.format(model_fpath))

    log_time(args.data_dpath, 'training', model_name, training_time)


if __name__ == '__main__':
    args = parse_args()
    #### Force using one GPU! That's a hack for FAISS, but we use it everywhere for consistency :) #####
    if args.gpu != -1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        args.gpu = 0

    device = get_device(args.gpu)
    train(args, device)
