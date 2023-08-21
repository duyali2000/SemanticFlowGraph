import argparse
import itertools
import json
import logging
import os
import shutil
import queue
import sys
import threading
import time
from datetime import timedelta

import numpy as np
import torch
import ujson

from SemanticCodebert import SemanticCodebert
from inference import ModelInference
from manager import IndexManager
from utils_colbert import create_directory

sys.path.append('.')
from utils import set_seed, get_device, log_time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('indexer')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', default=-1, type=int)

    # USE SAME VALUES AS FOR THE MODEL - MODEL NAME (--checkpoint) INCLUDES NECESSARY INFORMATION`
    parser.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'])
    parser.add_argument('--dim', dest='dim', default=256, type=int)
    parser.add_argument('--query_maxlen', dest='query_maxlen', default=256, type=int)
    parser.add_argument('--doc_maxlen', dest='doc_maxlen', default=256, type=int)
    parser.add_argument('--mask-punctuation', dest='mask_punctuation', default=True, action='store_true')

    parser.add_argument('--embeddings-comparison', choices=['average', 'token'], default='token')
    parser.add_argument('--checkpoint', dest='checkpoint',
                        default='../../../data/zxing/model_ColBERT_zxing_hunks_bertoverflow_QARCL_q256_d256_dim128_cosine_hunk')

    parser.add_argument('--bsize', dest='bsize', default=32, type=int)
    parser.add_argument('--amp', dest='amp', default=True, action='store_true')
    parser.add_argument('--data-dpath', default='../../../data/zxing')

    # DO NOT CHANGE THIS
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--nranks', default=1, type=int)
    parser.add_argument('--chunksize', dest='chunksize', default=6.0, required=False, type=float)  # in GiBs

    return parser.parse_args()


def main():
    set_seed(12345)

    args = parse_args()

    #### Force using one, selected GPU! #####
    if args.gpu != -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        args.gpu = 0

    args.granularity = args.checkpoint.split('_')[-1]

    index_root_dpath = os.path.join(args.data_dpath, 'index')
    index_name = get_index_name(args)
    index_dpath = os.path.join(index_root_dpath, index_name)

    #assert not os.path.exists(index_dpath), index_dpath
    if os.path.exists(index_dpath):
        shutil.rmtree(index_dpath)

    if args.rank < 1:
        create_directory(index_root_dpath)
        create_directory(index_dpath)

    logger.info('#> index_root = {0}'.format(index_root_dpath))

    process_idx = max(0, args.rank)
    encoder = CollectionEncoder(args, index_dpath, process_idx=process_idx, num_processes=args.nranks)

    t0 = time.time()
    encoder.encode()

    # Save metadata.
    if args.rank < 1:
        metadata_path = os.path.join(index_dpath, 'metadata.json')
        logger.info('Saving (the following) metadata to {0} ...'.format(metadata_path))
        logger.info(args)

        with open(metadata_path, 'w') as output_metadata:
            ujson.dump(args.__dict__, output_metadata)

    indexing_time = timedelta(seconds=time.time() - t0)
    log_time(args.data_dpath, 'indexing', index_name, indexing_time)


def get_index_name(args):
    model_fname = args.checkpoint.split('/')[-1]
    config = model_fname.split('_')[2:]
    print(config)
    project, dataset, model_config, tokens, query_maxl, doc_maxl, dim, similarity, granularity = config
    query_maxl = int(query_maxl[1:])
    doc_maxl = int(doc_maxl[1:])
    dim = int(dim[3:])
    if args.query_maxlen != query_maxl:
        logger.warning(
            'Maximum query length is different! Model trained with {0}. Inference with {1}.'.format(args.query_maxlen,
                                                                                                    query_maxl))
    if args.doc_maxlen != doc_maxl:
        logger.warning(
            'Maximum doc length is different! Model trained with {0}. Inference with {1}.'.format(args.doc_maxlen,
                                                                                                  doc_maxl))
    if args.dim != dim:
        logger.warning('Dimension is different! Model trained with {0}. Inference with {1}.'.format(args.dim, dim))

    name = 'INDEX_SemanticCodebert_' + '_'.join(
        [dataset, model_config, tokens, 'q' + str(query_maxl), 'd' + str(doc_maxl), 'dim' + str(dim), similarity,
         'q' + str(args.query_maxlen), 'd' + str(args.doc_maxlen), 'dim' + str(args.dim), granularity,
         args.embeddings_comparison])

    return name


class CollectionEncoder:
    def __init__(self, args, index_dpath, process_idx, num_processes):
        self.args = args
        self.index_dpath = index_dpath
        self.collection = os.path.join(args.data_dpath, 'doc_list_{0}.tsv'.format(args.granularity))
        self.process_idx = process_idx
        self.num_processes = num_processes

        assert 0.5 <= args.chunksize <= 128.0
        max_bytes_per_file = args.chunksize * (1024 * 1024 * 1024)

        max_bytes_per_doc = (self.args.doc_maxlen * self.args.dim * 2.0)

        # Determine subset sizes for output
        minimum_subset_size = 10_000
        maximum_subset_size = max_bytes_per_file / max_bytes_per_doc
        maximum_subset_size = max(minimum_subset_size, maximum_subset_size)
        self.possible_subset_sizes = [int(maximum_subset_size)]

        logger.info('#> Local args.bsize = {0}'.format(args.bsize))

        device = get_device(args.gpu)
        self._load_model(device)
        self.indexmgr = IndexManager(args.dim)

    def _saver_thread(self):
        for args in iter(self.saver_queue.get, None):
            self._save_batch(*args)

    def _load_model(self, device):
        self.colbert = load_colbert(self.args, device)
        self.colbert = self.colbert.to(device)

        self.inference = ModelInference(self.colbert, self.args, amp=self.args.amp)

    def encode(self):
        self.saver_queue = queue.Queue(maxsize=3)
        thread = threading.Thread(target=self._saver_thread)
        thread.start()

        t0 = time.time()
        local_docs_processed = 0

        with open(self.collection) as fi:
            for batch_idx, (offset, lines, owner) in enumerate(self._batch_passages(fi)):
                if owner != self.process_idx:
                    continue

                t1 = time.time()
                batch = self._preprocess_batch(offset, lines)
                embs, doclens = self._encode_batch(batch_idx, batch)

                t2 = time.time()
                self.saver_queue.put((batch_idx, embs, offset, doclens))

                t3 = time.time()
                local_docs_processed += len(lines)
                overall_throughput = compute_throughput(local_docs_processed, t0, t3)
                this_encoding_throughput = compute_throughput(len(lines), t1, t2)
                this_saving_throughput = compute_throughput(len(lines), t2, t3)

                logging.info('#> Completed batch #{0} (starting at passage #{1})'.format(batch_idx, offset))
                logging.info('#> Passages/min: {0} (overall); {1} (this encoding); {2} (this saving)'.
                             format(overall_throughput, this_encoding_throughput, this_saving_throughput))

            self.saver_queue.put(None)

        logger.info('#> Joining saver thread.')
        thread.join()

    def _batch_passages(self, fi):
        '''
        Must use the same seed across processes!
        '''
        np.random.seed(0)

        offset = 0
        for owner in itertools.cycle(range(self.num_processes)):
            batch_size = np.random.choice(self.possible_subset_sizes)

            L = [line for _, line in zip(range(batch_size), fi)]

            if len(L) == 0:
                break  # EOF

            yield (offset, L, owner)
            offset += len(L)

            if len(L) < batch_size:
                break  # EOF

        logger.info('[NOTE] Done with local share.')

        return

    def _preprocess_batch(self, offset, lines):
        endpos = offset + len(lines)

        batch = []

        for line_idx, line in zip(range(offset, endpos), lines):
            line_parts = line.strip().split('\t')
            pid, fpath = line_parts

            with open(os.path.join(self.args.data_dpath, self.args.granularity, fpath)) as f:
                data = json.load(f)
                passage = data['commit']

            assert len(passage) >= 1
            batch.append(passage)
            assert pid == 'id' or int(pid) == line_idx

        return batch

    def _encode_batch(self, batch_idx, batch):
        with torch.no_grad():
            embs = self.inference.docFromText(batch, bsize=self.args.bsize, keep_dims=False)
            assert type(embs) is list
            assert len(embs) == len(batch)

            local_doclens = [d.size(0) for d in embs]
            embs = torch.cat(embs)

        return embs, local_doclens

    def _save_batch(self, batch_idx, embs, offset, doclens):
        start_time = time.time()

        output_path = os.path.join(self.index_dpath, '{}.pt'.format(batch_idx))
        output_sample_path = os.path.join(self.index_dpath, '{}.sample'.format(batch_idx))
        doclens_path = os.path.join(self.index_dpath, 'doclens.{}.json'.format(batch_idx))

        # Save the embeddings.
        self.indexmgr.save(embs, output_path)
        self.indexmgr.save(embs[torch.randint(0, high=embs.size(0), size=(embs.size(0) // 20,))], output_sample_path)

        # Save the doclens.
        with open(doclens_path, 'w') as output_doclens:
            ujson.dump(doclens, output_doclens)

        throughput = compute_throughput(len(doclens), start_time, time.time())
        logger.info('#> Saved batch #{0} to {1} \t\t'.format(batch_idx, output_path))
        logger.info('Throughput = {0} passages per minute.'.format(throughput))


def compute_throughput(size, t0, t1):
    throughput = size / (t1 - t0) * 60

    if throughput > 1000 * 1000:
        throughput = throughput / (1000 * 1000)
        throughput = round(throughput, 1)
        return '{}M'.format(throughput)

    throughput = throughput / (1000)
    throughput = round(throughput, 1)
    return '{}k'.format(throughput)


def load_colbert(args, device):
    config = get_config(args)
    token_config = args.checkpoint.split('/')[-1].split('_')[5]
    if os.path.exists(args.checkpoint):
        logger.info('Loading model from {0}'.format(args.checkpoint))
        model = SemanticCodebert(config, token_config, dev=device, query_maxlen=args.query_maxlen, doc_maxlen=args.doc_maxlen,
                        dim=args.dim, similarity_metric=args.similarity, mask_punctuation=args.mask_punctuation)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        return model
    else:
        raise RuntimeError('Cannot load model from {0}. Path does not exist.'.format(args.checkpoint))


def get_config(args):
    if 'bertoverflow' in args.checkpoint:
        return '../BERTOverflow'
    elif 'bertbase' in args.checkpoint:
        return 'bert-base-uncased'
    elif 'codebert' in args.checkpoint:
        return 'microsoft/codebert-base'
    else:
        raise RuntimeError('Unknown BERT config in {0}'.format(args.checkpoint))


if __name__ == '__main__':
    main()
