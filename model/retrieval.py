import logging
import os
from collections import OrderedDict
from multiprocessing import Pool

import faiss
import torch

from faiss_indexer import load_doclens

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('retrieval')


def load_queries(queries_path):
    queries = OrderedDict()

    logger.info('#> Loading the queries from {0} ...'.format(queries_path))
    for qfile in os.listdir(os.path.join(queries_path, 'short')):
        with open(os.path.join(queries_path, 'short', qfile)) as f:
            short = ' '.join(f.readlines()).strip()
        with open(os.path.join(queries_path, 'long', qfile)) as f:
            long = ' '.join(f.readlines()).strip()

        query = short + ' ' + long
        qid = int(qfile[:-4])
        assert (qid not in queries), ('Query QID', qid, 'is repeated!')
        queries[qid] = query

    logger.info('#> Got {0} queries. All QIDs are unique'.format(len(queries)))

    return queries


def pid2commit_mapping(data_dpath, granularity):
    pid2commit = dict()
    collection = os.path.join(data_dpath, 'doc_list_{0}.tsv'.format(granularity))
    with open(collection, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            commit_file = line[1]
            commit_sha = commit_file.split('/')[-1][2:10]

            pid2commit[int(line[0])] = (commit_sha, commit_file)
    return pid2commit


def batch(group, bsize, provide_offset=False):
    offset = 0
    while offset < len(group):
        L = group[offset: offset + bsize]
        yield ((offset, L) if provide_offset else L)
        offset += len(L)


class FaissIndex:
    def __init__(self, args):
        logger.info('#> Loading the FAISS index from {0} ...'.format(args.faiss_index_path))

        faiss_part_range = os.path.basename(args.faiss_index_path).split('.')[-2].split('-')

        if len(faiss_part_range) == 2:
            faiss_part_range = range(*map(int, faiss_part_range))
            assert args.part_range[0] in faiss_part_range, (args.part_range, faiss_part_range)
            assert args.part_range[-1] in faiss_part_range, (args.part_range, faiss_part_range)
        else:
            faiss_part_range = None

        self.part_range = args.part_range
        self.faiss_part_range = faiss_part_range

        self.faiss_index = faiss.read_index(args.faiss_index_path)
        self.faiss_index.nprobe = args.nprobe

        logger.info('#> Building the emb2pid mapping..')
        all_doclens = load_doclens(args.index_path, flatten=False)

        pid_offset = 0
        if faiss_part_range is not None:
            print(f'#> Restricting all_doclens to the range {faiss_part_range}.')
            pid_offset = len(flatten(all_doclens[:faiss_part_range.start]))
            all_doclens = all_doclens[faiss_part_range.start:faiss_part_range.stop]

        self.relative_range = None
        if self.part_range is not None:
            start = self.faiss_part_range.start if self.faiss_part_range is not None else 0
            a = len(flatten(all_doclens[:self.part_range.start - start]))
            b = len(flatten(all_doclens[:self.part_range.stop - start]))
            self.relative_range = range(a, b)
            print(f'self.relative_range = {self.relative_range}')

        all_doclens = flatten(all_doclens)

        total_num_embeddings = sum(all_doclens)
        self.emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

        offset_doclens = 0
        for pid, dlength in enumerate(all_doclens):
            self.emb2pid[offset_doclens: offset_doclens + dlength] = pid_offset + pid
            offset_doclens += dlength

        logger.info('len(self.emb2pid) = {0}'.format(len(self.emb2pid)))

        self.parallel_pool = Pool(16)

        self.pid2commit = pid2commit_mapping(args.data_dpath, args.granularity)
        self.data_dpath = args.data_dpath
        self.granularity = args.granularity

    def retrieve(self, faiss_depth, Q, Q_ts, commits2ts):
        # embeddings ids are of size (#queries, max_query_length * faiss_depth)
        # data has been flattened
        embedding_ids, distances = self.queries_to_embedding_ids(faiss_depth, Q)
        pids_distances = self.embedding_ids_to_pids(embedding_ids, distances)
        pids_distances = self.filter_by_ts(pids_distances, Q_ts, commits2ts)
        return pids_distances

    def filter_by_ts(self, pids_distances, Q_ts, commits2ts):
        filtered_pids = list()
        filtered_distances = list()
        for idx, (pid, dist) in enumerate(zip(pids_distances[0][0], pids_distances[0][1])):
            short_sha, fname = self.pid2commit[pid]
            sha = fname.split('_')[1]
            ts = commits2ts[sha]
            if ts < Q_ts:
                filtered_pids.append(pid)
                filtered_distances.append(dist)
        logger.info(
            'Filtered {0} out of {1} because o timestamps.'.format(len(pids_distances[0][0]) - len(filtered_pids),
                                                                   len(pids_distances[0][0])))
        return [(filtered_pids, filtered_distances)]

    def queries_to_embedding_ids(self, faiss_depth, Q):
        # Flatten into a matrix for the faiss search.
        # Q is of shape (#queries, query_max_length, emb_size)
        num_queries, embeddings_per_query, dim = Q.size()
        # Q_faiss is of shape (#queries * query_max_length, emb_size)
        Q_faiss = Q.view(num_queries * embeddings_per_query, dim).cpu().contiguous()

        # Search in large batches with faiss.
        logger.info('#> Search in batches with faiss. Q.size={0}\tQ_faiss.size()={1}'.format(Q.size(), Q_faiss.size()))

        embeddings_ids = []
        distances = []
        faiss_bsize = embeddings_per_query * 5000
        for offset in range(0, Q_faiss.size(0), faiss_bsize):
            endpos = min(offset + faiss_bsize, Q_faiss.size(0))

            logger.info('#> Searching from {} to {}...'.format(offset, endpos))

            some_Q_faiss = Q_faiss[offset:endpos].float().numpy()
            some_distances, some_embedding_ids = self.faiss_index.search(some_Q_faiss, faiss_depth)
            embeddings_ids.append(torch.from_numpy(some_embedding_ids))
            distances.append(torch.from_numpy(some_distances))

        # embeding_ids are of shape (#queries * query_max_length, faiss_depth)
        # meaning for each (token) embedding we located faiss_depth the most similar token embeddings
        embedding_ids = torch.cat(embeddings_ids)

        # distances are of shape (#queries * query_max_length, faiss_depth)
        # meaning for each (token) embedding we know a distance to token embedding as indicated by embeddings_ids
        distances = torch.cat(distances)

        # Reshape to (number of queries, non-unique embedding IDs per query)
        embedding_ids = embedding_ids.view(num_queries, embeddings_per_query * embedding_ids.size(1))
        distances = distances.view(num_queries, embeddings_per_query * distances.size(1))
        return embedding_ids, distances

    def embedding_ids_to_pids(self, embedding_ids, distances):
        # Find unique PIDs per query.
        logger.info('#> Lookup the PIDs..')
        all_pids = self.emb2pid[embedding_ids]

        logger.info('#> Converting to a list [shape = {0}]..'.format(all_pids.size()))
        all_pids = all_pids.tolist()
        distances = distances.tolist()

        logger.info('#> Removing duplicates (in parallel if large enough)..')
        data = list()
        for idx, qpid in enumerate(all_pids):
            data.append(list(zip(qpid, distances[idx])))

        if len(all_pids) > 5000:
            pids_distances = list(self.parallel_pool.map(uniq, all_pids, distances))
        else:
            pids_distances = list(map(uniq, all_pids, distances))

        logger.info('#> Done with embedding_ids_to_pids()')

        return pids_distances


def flatten(L):
    return [x for y in L for x in y]


def uniq(pids, distances):
    new_l = list()
    new_d = list()
    added_pids = set()
    for idx, pid in enumerate(pids):
        if pid in added_pids:
            continue
        added_pids.add(pid)
        new_l.append(pid)
        new_d.append(distances[idx])
    return new_l, new_d
