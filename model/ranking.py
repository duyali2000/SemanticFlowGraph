import argparse
import itertools
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
from itertools import accumulate

import torch

from faiss_indexer import load_doclens, get_parts, load_index_part, get_faiss_index_name
from indexer import load_colbert
from inference import ModelInference
from manager import NullContextManager
from retrieval import load_queries, FaissIndex, flatten
from retrieval import pid2commit_mapping

sys.path.append('.')
from utils import set_seed, get_device, log_time
from utils_colbert import load_br2ts, load_commit2ts

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('ranking')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'])
    parser.add_argument('--dim', dest='dim', default=128, type=int)
    parser.add_argument('--query-maxlen', dest='query_maxlen', default=256, type=int)
    parser.add_argument('--doc-maxlen', dest='doc_maxlen', default=230, type=int)
    parser.add_argument('--mask-punctuation', dest='mask_punctuation', default=True, action='store_true')

    parser.add_argument('--checkpoint', dest='checkpoint',
                        default='../../../data/zxing/model_SemanticCodebert_zxing_RN_bertoverflow_QARCL_q256_d230_dim128_cosine_commits')
    parser.add_argument('--bsize', dest='bsize', default=4, type=int)
    parser.add_argument('--amp', dest='amp', default=True, action='store_true')

    parser.add_argument('--data-dpath', default='../../../data/zxing')
    parser.add_argument('--index_name', dest='index_name',
                        default='INDEX_SemanticCodebert_RN_bertoverflow_QARCL_q256_d230_dim128_cosine_q256_d230_dim128_commits_token')
    parser.add_argument('--nprobe', dest='nprobe', default=100, type=int)

    parser.add_argument('--faiss_name', dest='faiss_name', default='ivfpq.320.faiss', type=str)
    parser.add_argument('--faiss_depth', dest='faiss_depth', default=1024, type=int,
                        help='Number of documents returned via faiss.search')
    parser.add_argument('--part-range', dest='part_range', default=None, type=str)
    parser.add_argument('--depth', dest='depth', default=1000, type=int)

    return parser.parse_args()


def main():
    set_seed(12345)
    args = parse_args()

    # fix for FAISS
    if args.gpu != -1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        args.gpu = 0

    args.depth = args.depth if args.depth > 0 else None
    device = get_device(args.gpu)
    args.granularity = args.checkpoint.split('_')[-1]

    if args.part_range:
        part_offset, part_endpos = map(int, args.part_range.split('..'))
        args.part_range = range(part_offset, part_endpos)

    args.colbert = load_colbert(args, device).to(device)
    args.queries = load_queries(os.path.join(args.data_dpath, 'br'))
    args.embeddings_comparison = args.index_name.split('_')[-1]

    args.index_path = os.path.join(args.data_dpath, 'index', args.index_name)

    if args.faiss_name is not None:
        args.faiss_index_path = os.path.join(args.index_path, args.faiss_name)
    else:
        args.faiss_index_path = os.path.join(args.index_path, get_faiss_index_name(args))

    t0 = time.time()
    retrieve(args, device)

    ranking_time = timedelta(seconds=time.time() - t0)
    log_time(args.data_dpath, 'ranking', args.index_name, ranking_time)


def retrieve(args, device):
    inference = ModelInference(args.colbert, args, amp=args.amp)
    ranker = Ranker(args, inference, device, faiss_depth=args.faiss_depth)

    fpath_ranking = 'ranking_' + args.index_name + '.tsv'
    granularity = args.index_name.split('_')[-2]
    ranking_logger = RankingLogger(args.data_dpath, granularity, qrels=None)
    bug2ts = load_br2ts(args.data_dpath)
    commits2ts = load_commit2ts(args.data_dpath)
    with ranking_logger.context(fpath_ranking, also_save_annotations=False) as rlogger:
        queries = args.queries
        qids_in_order = list(queries.keys())

        for qoffset, qbatch in batch(qids_in_order, 100, provide_offset=True):
            qbatch_text = [queries[qid] for qid in qbatch]

            rankings = []

            for query_idx, q in enumerate(qbatch_text):
                if 'cuda' in device.type:
                    torch.cuda.synchronize(device)
                s = time.time()

                Q, attention_mask = ranker.encode([q])
                pids, scores = ranker.rank(Q, bug2ts[qbatch[query_idx]], commits2ts)

                if 'cuda' in device.type:
                    torch.cuda.synchronize()
                milliseconds = (time.time() - s) * 1000.0
                logger.info('#> Processing of query {0} took {1} ms'.format(qbatch[query_idx], milliseconds))
                rankings.append(zip(pids, scores))

            for query_idx, (qid, ranking) in enumerate(zip(qbatch, rankings)):
                query_idx = qoffset + query_idx

                if query_idx % 100 == 0:
                    logger.info('#> Logging query #{0} (qid {1}) now...'.format(query_idx, qid))

                ranking = [(score, pid, None) for pid, score in itertools.islice(ranking, args.depth)]
                rlogger.log(qid, ranking, is_ranked=True)

    logger.info(ranking_logger.filename)
    logger.info('#> Done.')


def batch(group, bsize, provide_offset=False):
    offset = 0
    while offset < len(group):
        L = group[offset: offset + bsize]
        yield ((offset, L) if provide_offset else L)
        offset += len(L)
    return


class Ranker:
    def __init__(self, args, inference, device, faiss_depth=1024):
        self.inference = inference
        self.faiss_depth = faiss_depth

        if faiss_depth is not None:
            self.faiss_index = FaissIndex(args)
            self.retrieve = partial(self.faiss_index.retrieve, self.faiss_depth)

        self.index = IndexPart(args.index_path, device, dim=inference.colbert.dim, part_range=args.part_range,
                               verbose=True)

    def encode(self, queries):
        assert type(queries) in [list, tuple], type(queries)

        Q, attention_mask = self.inference.queryFromText(queries, bsize=512 if len(queries) > 512 else None)

        return Q, attention_mask

    def rank(self, Q, Q_ts, commits2ts, pids=None):
        pids, _ = self.retrieve(Q, Q_ts, commits2ts)[0] if pids is None else pids

        assert type(pids) in [list, tuple], type(pids)
        assert Q.size(0) == 1, (len(pids), Q.size())
        assert all(type(pid) is int for pid in pids)

        scores = []
        if len(pids) > 0:
            Q = Q.permute(0, 2, 1)
            scores = self.index.rank(Q, pids)

            scores_sorter = torch.tensor(scores).sort(descending=True)
            pids, scores = torch.tensor(pids)[scores_sorter.indices].tolist(), scores_sorter.values.tolist()

        return pids, scores


class RankingLogger:
    def __init__(self, directory, granularity, qrels=None, log_scores=False):
        self.directory = directory
        self.qrels = qrels
        self.filename, self.also_save_annotations = None, None
        self.log_scores = log_scores
        self.pid2commit = pid2commit_mapping(self.directory, granularity)

    @contextmanager
    def context(self, filename, also_save_annotations=False):
        assert self.filename is None
        assert self.also_save_annotations is None

        results_dpath = os.path.join(self.directory, 'results')
        if not os.path.exists(results_dpath):
            os.makedirs(results_dpath)
        filename = os.path.join(results_dpath, filename)
        self.filename, self.also_save_annotations = filename, also_save_annotations

        logger.info('#> Logging ranked lists to {0}'.format(self.filename))

        with open(filename, 'w') as f:
            self.f = f
            with (open(filename + '.annotated', 'w') if also_save_annotations else NullContextManager()) as g:
                self.g = g
                try:
                    yield self
                finally:
                    pass

    def log(self, qid, ranking, is_ranked=True, print_positions=[]):
        print_positions = set(print_positions)

        f_buffer = []
        g_buffer = []

        for rank, (score, pid, passage) in enumerate(ranking):
            is_relevant = self.qrels and int(pid in self.qrels[qid])
            rank = rank + 1 if is_ranked else -1

            commit, file = self.pid2commit[pid]
            f_buffer.append('\t'.join([str(x) for x in [qid, pid, commit, file, rank, score]]) + '\n')
            if self.g:
                g_buffer.append('\t'.join([str(x) for x in [qid, pid, rank, is_relevant]]) + '\n')

            if rank in print_positions:
                prefix = '** ' if is_relevant else ''
                prefix += str(rank)
                logger.info('#> ( QID {} ) '.format(qid) + prefix + ') ', pid, ':', score, '    ', passage)

        self.f.write(''.join(f_buffer))
        if self.g:
            self.g.write(''.join(g_buffer))


class IndexPart:
    def __init__(self, directory, device, dim=128, part_range=None, verbose=True):
        first_part, last_part = (0, None) if part_range is None else (part_range.start, part_range.stop)

        # Load parts metadata
        all_parts, all_parts_paths, _ = get_parts(directory)
        self.parts = all_parts[first_part:last_part]
        self.parts_paths = all_parts_paths[first_part:last_part]

        # Load doclens metadata
        all_doclens = load_doclens(directory, flatten=False)

        self.doc_offset = sum([len(part_doclens) for part_doclens in all_doclens[:first_part]])
        self.doc_endpos = sum([len(part_doclens) for part_doclens in all_doclens[:last_part]])
        self.pids_range = range(self.doc_offset, self.doc_endpos)

        self.parts_doclens = all_doclens[first_part:last_part]
        self.doclens = flatten(self.parts_doclens)
        self.num_embeddings = sum(self.doclens)

        self.tensor = self._load_parts(dim, verbose)
        self.ranker = IndexRanker(self.tensor, self.doclens, device)

    def _load_parts(self, dim, verbose):
        tensor = torch.zeros(self.num_embeddings + 512, dim, dtype=torch.float16)

        if verbose:
            logger.info("tensor.size() = {0}".format(tensor.size()))

        offset = 0
        for idx, filename in enumerate(self.parts_paths):
            logger.info("|> Loading {0} ...".format(filename))

            endpos = offset + sum(self.parts_doclens[idx])
            part = load_index_part(filename)

            tensor[offset:endpos] = part
            offset = endpos

        return tensor

    def pid_in_range(self, pid):
        return pid in self.pids_range

    def rank(self, Q, pids):
        """
        Rank a single batch of Q x pids (e.g., 1k--10k pairs).
        """

        assert Q.size(0) in [1, len(pids)], (Q.size(0), len(pids))
        assert all(pid in self.pids_range for pid in pids), self.pids_range

        pids_ = [pid - self.doc_offset for pid in pids]
        scores = self.ranker.rank(Q, pids_)

        return scores

    def batch_rank(self, all_query_embeddings, query_indexes, pids, sorted_pids):
        """
        Rank a large, fairly dense set of query--passage pairs (e.g., 1M+ pairs).
        Higher overhead, much faster for large batches.
        """

        assert ((pids >= self.pids_range.start) & (pids < self.pids_range.stop)).sum() == pids.size(0)

        pids_ = pids - self.doc_offset
        scores = self.ranker.batch_rank(all_query_embeddings, query_indexes, pids_, sorted_pids)

        return scores


BSIZE = 1 << 14


class IndexRanker:
    def __init__(self, tensor, doclens, device):
        self.tensor = tensor.to(device)
        self.doclens = doclens
        self.device = device

        self.maxsim_dtype = torch.float32
        self.doclens_pfxsum = [0] + list(accumulate(self.doclens))

        self.doclens = torch.tensor(self.doclens)
        self.doclens_pfxsum = torch.tensor(self.doclens_pfxsum)

        self.dim = self.tensor.size(-1)

        self.strides = [torch_percentile(self.doclens, p) for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]]
        self.strides.append(self.doclens.max().item())
        self.strides = sorted(list(set(self.strides)))

        logger.info("#> Using strides {0}..".format(self.strides))

        self.views = self._create_views(self.tensor)
        self.buffers = self._create_buffers(BSIZE, self.tensor.dtype, [device])

    def _create_views(self, tensor):
        views = []

        for stride in self.strides:
            outdim = tensor.size(0) - stride + 1
            view = torch.as_strided(tensor, (outdim, stride, self.dim), (self.dim, self.dim, 1))
            views.append(view)

        return views

    def _create_buffers(self, max_bsize, dtype, devices):
        buffers = {}

        for device in devices:
            buffers[device] = [
                torch.zeros(max_bsize, stride, self.dim, dtype=dtype, device=device)
                for stride in self.strides]

        return buffers

    def rank(self, Q, pids, views=None, shift=0):
        assert len(pids) > 0
        assert Q.size(0) in [1, len(pids)]

        Q = Q.contiguous().to(self.device).to(dtype=self.maxsim_dtype)

        views = self.views if views is None else views
        VIEWS_DEVICE = views[0].device

        D_buffers = self.buffers[VIEWS_DEVICE]

        raw_pids = pids if type(pids) is list else pids.tolist()
        pids = torch.tensor(pids) if type(pids) is list else pids

        doclens, offsets = self.doclens[pids], self.doclens_pfxsum[pids]

        assignments = (doclens.unsqueeze(1) > torch.tensor(self.strides).unsqueeze(0) + 1e-6).sum(-1)

        one_to_n = torch.arange(len(raw_pids))
        output_pids, output_scores, output_permutation = [], [], []

        for group_idx, stride in enumerate(self.strides):
            locator = (assignments == group_idx)

            if locator.sum() < 1e-5:
                continue

            group_pids, group_doclens, group_offsets = pids[locator], doclens[locator], offsets[locator]
            group_Q = Q if Q.size(0) == 1 else Q[locator]

            group_offsets = group_offsets.to(VIEWS_DEVICE) - shift
            group_offsets_uniq, group_offsets_expand = torch.unique_consecutive(group_offsets, return_inverse=True)

            D_size = group_offsets_uniq.size(0)
            D = torch.index_select(views[group_idx], 0, group_offsets_uniq, out=D_buffers[group_idx][:D_size])
            D = D.to(self.device)
            D = D[group_offsets_expand.to(self.device)].to(dtype=self.maxsim_dtype)

            mask = torch.arange(stride, device=self.device) + 1
            mask = mask.unsqueeze(0) <= group_doclens.to(self.device).unsqueeze(-1)

            scores = (D @ group_Q) * mask.unsqueeze(-1)
            scores = scores.max(1).values.sum(-1).cpu()
            output_pids.append(group_pids)
            output_scores.append(scores)
            output_permutation.append(one_to_n[locator])

        output_permutation = torch.cat(output_permutation).sort().indices
        output_pids = torch.cat(output_pids)[output_permutation].tolist()
        output_scores = torch.cat(output_scores)[output_permutation].tolist()

        assert len(raw_pids) == len(output_pids)
        assert len(raw_pids) == len(output_scores)
        assert raw_pids == output_pids

        return output_scores

    def batch_rank(self, all_query_embeddings, all_query_indexes, all_pids, sorted_pids):
        assert sorted_pids is True

        ######

        scores = []
        range_start, range_end = 0, 0

        for pid_offset in range(0, len(self.doclens), 50_000):
            pid_endpos = min(pid_offset + 50_000, len(self.doclens))

            range_start = range_start + (all_pids[range_start:] < pid_offset).sum()
            range_end = range_end + (all_pids[range_end:] < pid_endpos).sum()

            pids = all_pids[range_start:range_end]
            query_indexes = all_query_indexes[range_start:range_end]

            logger.info("###--> Got {0} query--passage pairs in this sub-range [{1}-{2}]".format(len(pids), pid_offset,
                                                                                                 pid_endpos))

            if len(pids) == 0:
                continue

            logger.info("###--> Ranking in batches the pairs #{0} through #{1} in this sub-range".format(range_start,
                                                                                                         range_end))

            tensor_offset = self.doclens_pfxsum[pid_offset].item()
            tensor_endpos = self.doclens_pfxsum[pid_endpos].item() + 512

            collection = self.tensor[tensor_offset:tensor_endpos].to(self.device)
            views = self._create_views(collection)

            logger.info("#> Ranking in batches of {0} query--passage pairs...".format(BSIZE))

            for batch_idx, offset in enumerate(range(0, len(pids), BSIZE)):
                if batch_idx % 100 == 0:
                    logger.info("#> Processing batch #{}..".format(batch_idx))

                endpos = offset + BSIZE
                batch_query_index, batch_pids = query_indexes[offset:endpos], pids[offset:endpos]

                Q = all_query_embeddings[batch_query_index]

                scores.extend(self.rank(Q, batch_pids, views, shift=tensor_offset))

        return scores


def torch_percentile(tensor, p):
    assert p in range(1, 100 + 1)
    assert tensor.dim() == 1

    return tensor.kthvalue(int(p * tensor.size(0) / 100.0)).values.item()


if __name__ == '__main__':
    main()
