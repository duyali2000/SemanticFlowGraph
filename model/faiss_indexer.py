import argparse
import itertools
import logging
import math
import os
import queue
import sys
import threading
import time
from datetime import timedelta

import faiss
import numpy as np
import torch
import ujson

sys.path.append('.')
from utils import set_seed, log_time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('faiss-index')

SPAN = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--data-dpath', default='../../../data/zxing')
    parser.add_argument('--index-name', dest='index_name',
                        default='INDEX_SemanticCodebert_RN_bertoverflow_QARCL_q256_d230_dim128_cosine_q256_d230_dim128_commits_token')

    parser.add_argument('--partitions', dest='partitions', default=320, type=int)
    parser.add_argument('--sample', dest='sample', default=0.5, type=float)
    parser.add_argument('--slices', dest='slices', default=1, type=int)
    return parser.parse_args()


def main():
    set_seed(12345)

    args = parse_args()
    print(type(args.gpu))
    #### Force using one GPU! #####
    if args.gpu != -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        args.gpu = -1
    print(args.gpu)
    assert args.slices >= 1
    assert args.sample is None or (0.0 < args.sample < 1.0), args.sample

    args.index_path = os.path.join(args.data_dpath, 'index', args.index_name)
    assert os.path.exists(args.index_path), args.index_path

    num_embeddings = sum(load_doclens(args.index_path))
    logger.info('#> num_embeddings = {0}'.format(num_embeddings))

    if args.partitions is None:
        args.partitions = 1 << math.ceil(math.log2(8 * math.sqrt(num_embeddings)))
        logger.warning('You did not specify --partitions!')
        logger.warning(
            'Default computation chooses {0} partitions (for {1} embeddings)'.format(args.partitions, num_embeddings))

    t0 = time.time()
    print(args.gpu)
    index_faiss(args)

    faiss_indexing_time = timedelta(seconds=time.time() - t0)
    log_time(args.data_dpath, 'faiss', args.index_name, faiss_indexing_time)


def index_faiss(args):
    logger.info('#> Starting..')

    parts, parts_paths, samples_paths = get_parts(args.index_path)

    if args.sample is not None:
        assert args.sample, args.sample
        logger.info(f'#> Training with {round(args.sample * 100.0, 1)}% of *all* embeddings (provided --sample).')
        samples_paths = parts_paths

    num_parts_per_slice = math.ceil(len(parts) / args.slices)

    for slice_idx, part_offset in enumerate(range(0, len(parts), num_parts_per_slice)):
        part_endpos = min(part_offset + num_parts_per_slice, len(parts))

        slice_parts_paths = parts_paths[part_offset:part_endpos]
        slice_samples_paths = samples_paths[part_offset:part_endpos]

        if args.slices == 1:
            faiss_index_name = get_faiss_index_name(args)
        else:
            faiss_index_name = get_faiss_index_name(args, offset=part_offset, endpos=part_endpos)

        output_path = os.path.join(args.index_path, faiss_index_name)
        logger.info(f'#> Processing slice #{slice_idx + 1} of {args.slices} (range {part_offset}..{part_endpos}).')
        logger.info(f'#> Will write to {output_path}.')

        assert not os.path.exists(output_path), output_path

        index = prepare_faiss_index(slice_samples_paths, args.partitions, args.gpu, args.sample)

        loaded_parts = queue.Queue(maxsize=1)

        def _loader_thread(thread_parts_paths):
            for filenames in grouper(thread_parts_paths, SPAN, fillvalue=None):
                sub_collection = [load_index_part(filename) for filename in filenames if filename is not None]
                sub_collection = torch.cat(sub_collection)
                sub_collection = sub_collection.float().cpu().numpy()
                loaded_parts.put(sub_collection)

        thread = threading.Thread(target=_loader_thread, args=(slice_parts_paths,))
        thread.start()

        logger.info('#> Indexing the vectors...')

        for filenames in grouper(slice_parts_paths, SPAN, fillvalue=None):
            logger.info('#> Loading {0} (from queue)...'.format(filenames))
            sub_collection = loaded_parts.get()

            logger.info('#> Processing a sub_collection with shape {0}'.format(sub_collection.shape))
            index.add(sub_collection)

        logger.info('Done indexing!')

        index.save(output_path)

        logger.info('Done! All complete (for slice #{0} of {1})!'.format(slice_idx + 1, args.slices))

        thread.join()


def get_parts(directory):
    extension = '.pt'

    parts = sorted([int(filename[: -1 * len(extension)]) for filename in os.listdir(directory)
                    if filename.endswith(extension)])

    assert list(range(len(parts))) == parts, parts

    # Integer-sortedness matters.
    parts_paths = [os.path.join(directory, '{}{}'.format(filename, extension)) for filename in parts]
    samples_paths = [os.path.join(directory, '{}.sample'.format(filename)) for filename in parts]

    return parts, parts_paths, samples_paths


def get_faiss_index_name(args, offset=None, endpos=None):
    partitions_info = '' if args.partitions is None else f'.{args.partitions}'
    range_info = '' if offset is None else f'.{offset}-{endpos}'

    return f'ivfpq{partitions_info}{range_info}.faiss'


def prepare_faiss_index(slice_samples_paths, partitions, gpu, sample_fraction=None):
    training_sample = load_sample(slice_samples_paths, sample_fraction=sample_fraction)

    dim = training_sample.shape[-1]
    index = FaissIndex(dim, partitions, gpu)

    logger.info('#> Training with the vectors...')

    index.train(training_sample)

    logger.info('Done training!\n')

    return index


def grouper(iterable, n, fillvalue=None):
    '''
    Collect data into fixed-length chunks or blocks
        Example: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx'
        Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
    '''

    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def load_index_part(filename):
    part = torch.load(filename)

    if type(part) == list:  # for backward compatibility
        part = torch.cat(part)

    return part


class FaissIndex:
    def __init__(self, dim, partitions, gpu):
        self.dim = dim
        self.partitions = partitions
        print(gpu)
        self.gpu = FaissIndexGPU(gpu)
        self.quantizer, self.index = self._create_index()
        self.offset = 0

    def _create_index(self):
        quantizer = faiss.IndexFlatL2(self.dim)  # faiss.IndexHNSWFlat(dim, 32)
        index = faiss.IndexIVFPQ(quantizer, self.dim, self.partitions, 16, 8)

        return quantizer, index

    def train(self, train_data):
        logger.info(f'#> Training now (using {self.gpu.ngpu} GPUs)...')

        if self.gpu.ngpu > 0:
            self.gpu.training_initialize(self.index, self.quantizer)

        s = time.time()
        self.index.train(train_data)
        print(time.time() - s)

        if self.gpu.ngpu > 0:
            self.gpu.training_finalize()

    def add(self, data):
        logger.info(f'Add data with shape {data.shape} (offset = {self.offset})..')

        if self.gpu.ngpu > 0 and self.offset == 0:
            self.gpu.adding_initialize(self.index)

        if self.gpu.ngpu > 0:
            self.gpu.add(self.index, data, self.offset)
        else:
            self.index.add(data)

        self.offset += data.shape[0]

    def save(self, output_path):
        logger.info(f'Writing index to {output_path} ...')

        self.index.nprobe = 10  # just a default
        faiss.write_index(self.index, output_path)


def load_doclens(directory, flatten=True):
    parts, _, _ = get_parts(directory)

    doclens_filenames = [os.path.join(directory, 'doclens.{}.json'.format(filename)) for filename in parts]
    all_doclens = [ujson.load(open(filename)) for filename in doclens_filenames]

    if flatten:
        all_doclens = [x for sub_doclens in all_doclens for x in sub_doclens]

    return all_doclens


def load_sample(samples_paths, sample_fraction=None):
    sample = []

    for filename in samples_paths:
        logger.info(f'#> Loading {filename} ...')
        part = load_index_part(filename)
        if sample_fraction:
            part = part[torch.randint(0, high=part.size(0), size=(int(part.size(0) * sample_fraction),))]
        sample.append(part)

    sample = torch.cat(sample).float().cpu().numpy()

    logger.info('#> Sample has shape {0}'.format(sample.shape))

    return sample


class FaissIndexGPU:
    def __init__(self, gpu):
        self.ngpu = 0 if gpu == -1 else 1
        self.gpu = 0

        if self.ngpu == 0:
            return

        self.tempmem = 1 << 33
        self.max_add_per_gpu = 1 << 25
        self.max_add = self.max_add_per_gpu * self.ngpu
        self.add_batch_size = 65536

        self.gpu_resources = self._prepare_gpu_resources()

    def _prepare_gpu_resources(self):
        logger.info(f'Preparing resources for {self.ngpu} GPUs.')

        gpu_resources = []

        for _ in range(self.ngpu):
            res = faiss.StandardGpuResources()
            if self.tempmem >= 0:
                res.setTempMemory(self.tempmem)
            gpu_resources.append(res)

        return gpu_resources

    def training_initialize(self, index, quantizer):
        '''
        The index and quantizer should be owned by caller.
        '''

        assert self.ngpu > 0

        s = time.time()
        self.index_ivf = faiss.extract_index_ivf(index)
        self.clustering_index = faiss.index_cpu_to_gpu(self.gpu_resources[0], self.gpu, quantizer)
        self.index_ivf.clustering_index = self.clustering_index
        print(time.time() - s)

    def training_finalize(self):
        assert self.ngpu > 0

        s = time.time()
        self.index_ivf.clustering_index = faiss.index_gpu_to_cpu(self.index_ivf.clustering_index)
        print(time.time() - s)

    def adding_initialize(self, index):
        '''
        The index should be owned by caller.
        '''

        assert self.ngpu > 0

        self.co = faiss.GpuMultipleClonerOptions()
        self.co.useFloat16 = True
        self.co.useFloat16CoarseQuantizer = False
        self.co.usePrecomputed = False
        self.co.indicesOptions = faiss.INDICES_CPU
        self.co.verbose = True
        self.co.reserveVecs = self.max_add
        self.co.shard = True
        assert self.co.shard_type in (0, 1, 2)

        self.gpu_index, self.vres, self.vdev = self.index_cpu_to_gpu_multiple(self.gpu_resources, index, self.co,
                                                                              gpu_nos=[self.gpu])

    def index_cpu_to_gpu_multiple(self, resources, index, co=None, gpu_nos=None):
        '''
        Custom cpu_to_gpu function to make sure we use only one, selected gpu
        '''

        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        if gpu_nos is None:
            gpu_nos = range(len(resources))
        for i, res in zip(gpu_nos, resources):
            vdev.push_back(i)
            vres.push_back(res)
        index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
        index.referenced_objects = resources
        return index, vres, vdev

    def add(self, index, data, offset):
        assert self.ngpu > 0

        t0 = time.time()
        nb = data.shape[0]

        for i0 in range(0, nb, self.add_batch_size):
            i1 = min(i0 + self.add_batch_size, nb)
            xs = data[i0:i1]

            self.gpu_index.add_with_ids(xs, np.arange(offset + i0, offset + i1))

            if self.max_add > 0 and self.gpu_index.ntotal > self.max_add:
                self._flush_to_cpu(index, nb, offset)

            print('\r%d/%d (%.3f s)  ' % (i0, nb, time.time() - t0), end=' ')
            sys.stdout.flush()

        if self.gpu_index.ntotal > 0:
            self._flush_to_cpu(index, nb, offset)

        assert index.ntotal == offset + nb, (index.ntotal, offset + nb, offset, nb)
        print(f'add(.) time: %.3f s \t\t--\t\t index.ntotal = {index.ntotal}' % (time.time() - t0))

    def _flush_to_cpu(self, index, nb, offset):
        print('Flush indexes to CPU')

        for i in range(self.ngpu):
            index_src_gpu = faiss.downcast_index(self.gpu_index if self.ngpu == 1 else self.gpu_index.at(i))
            index_src = faiss.index_gpu_to_cpu(index_src_gpu)

            # index_src.copy_subset_to(index, 0, 0, nb)  # original
            index_src.copy_subset_to(index, 0, offset, offset + nb)
            index_src_gpu.reset()
            index_src_gpu.reserveMemory(self.max_add)

        if self.ngpu > 1:
            self.gpu_index.sync_with_shard_indexes()


if __name__ == '__main__':
    main()
