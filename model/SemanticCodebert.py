import json
import logging
import os
import random
import string
from functools import partial

import torch
from torch import nn
from transformers import BertModel, BertTokenizerFast, RobertaModel, RobertaTokenizer, AutoTokenizer, AutoModel
from tokenizer import DocTokenizer, QueryTokenizer, tensorize_triples
from transformers import logging as log
log.set_verbosity_error()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('bert')
random.seed(1234)


class EagerBatcher:
    def __init__(self, args, rank=0, nranks=1):
        self.rank, self.nranks = rank, nranks
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.config, args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.config, args.doc_maxlen, args.special_tokens)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.data_dpath = args.data_dpath
        self.granularity = args.granularity

        self.triples_path = os.path.join(args.data_dpath, args.triples)
        self._reset_triples()

    def _reset_triples(self):
        cwd = os.getcwd()
        print(cwd)
        self.reader = open(self.triples_path, mode='r', encoding='utf-8')
        # skip header
        self.reader.readline()

        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        queries, positive_hunk, negative_hunk = [], [], []
        line_idx = 0
        for line_idx, line in zip(range(self.bsize * self.nranks * 2), self.reader):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            query, hunk, label = line.strip().split(',')
            with open(os.path.join(self.data_dpath, self.granularity, hunk), 'r') as f:
                data = json.load(f)
                hunk = data['commit']

            if label == '1.0':
                queries.append(query)
                positive_hunk.append(hunk)
            else:
                negative_hunk.append(hunk)

        self.position += line_idx + 1

        if len(queries) < self.bsize:
            raise StopIteration

        return self.collate(queries, positive_hunk, negative_hunk)

    def collate(self, queries, positive_hunk, negative_hunk):
        assert len(queries) == len(positive_hunk) == len(negative_hunk) == self.bsize

        return self.tensorize_triples(queries, positive_hunk, negative_hunk, self.bsize // self.accumsteps)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        self._reset_triples()

        logger.warning('Skipping to batch #{0} (with intended_batch_size = {1}) for training.'.format(batch_idx,
                                                                                                      intended_batch_size))

        _ = [self.reader.readline() for _ in range(batch_idx * intended_batch_size)]

        return None


class SemanticCodebert(nn.Module):
    def __init__(self, config, token_config, dev, query_maxlen=256, doc_maxlen=256, mask_punctuation=True, dim=128,
                 similarity_metric='cosine'):

        super(SemanticCodebert, self).__init__()
        self.config = config
        self.token_config = token_config
        assert token_config in ['QD', 'QARC', 'QARCL']
        self.bert_q_text = self._load_pretrained_bert()
        self.bert_k_text = self._load_pretrained_bert()
        self.bert_q_code = self._load_pretrained_bert_code()
        self.bert_k_code = self._load_pretrained_bert_code()

        self.dev = dev
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation is True:
            if 'codebert' in config:
                self.tokenizer = RobertaTokenizer.from_pretrained(config)
            else:
                self.tokenizer = BertTokenizerFast.from_pretrained(config)
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.dropout = torch.nn.Dropout(0.1)
        # this create lower-dim representation == lower memory consumption!
        self.linear_q_text = nn.Sequential(
            nn.Linear(self.bert_q_text.config.hidden_size * 4, self.bert_q_text.config.hidden_size * 2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(query_maxlen),
            nn.Linear(self.bert_q_text.config.hidden_size * 2, dim, bias=False)
                )
        self.linear_k_text = nn.Sequential(
            nn.Linear(self.bert_k_text.config.hidden_size * 4, self.bert_k_text.config.hidden_size * 2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(query_maxlen),
            nn.Linear(self.bert_k_text.config.hidden_size * 2, dim, bias=False)
                )

        self.linear_q_code = nn.Sequential(
            nn.Linear(self.bert_q_code.config.hidden_size * 4, self.bert_q_code.config.hidden_size * 2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(doc_maxlen),
            nn.Linear(self.bert_q_code.config.hidden_size * 2, dim, bias=False)
        )
        self.linear_k_code = nn.Sequential(
            nn.Linear(self.bert_k_code.config.hidden_size * 4, self.bert_k_code.config.hidden_size * 2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(doc_maxlen),
            nn.Linear(self.bert_k_code.config.hidden_size * 2, dim, bias=False)
        )

        self.K = 2048 #negatives
        self.m = 0.999
        self.T = 0.07

        # create the queue
        self.register_buffer("queue", torch.randn(dim, self.K))
        self.queue_query = nn.functional.normalize(self.queue, dim=0)
        self.queue_doc = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.model_pairs = [
            [self.bert_q_text, self.bert_k_text],
            [self.bert_q_code, self.bert_k_code],
            [self.linear_q_code, self.linear_k_code],
            [self.linear_q_text, self.linear_k_text],
        ]
        self.copy_params()

    # utils
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
            Momentum update of the key encoder
        """
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.m + param.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_query(self, keys):
        # gather keys before updating queue

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_doc[:, ptr:ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_doc(self, keys):
        # gather keys before updating queue

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_doc[:, ptr:ptr + batch_size] = keys.T

        #ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def _load_pretrained_bert(self):
        if 'codebert' in self.config:
            tokenizer = RobertaTokenizer.from_pretrained(self.config)
            bert = RobertaModel.from_pretrained(self.config)
            bert.resize_token_embeddings(len(tokenizer) + 4)
        else:
            bert = BertModel.from_pretrained(self.config)
        return bert

    def _load_pretrained_bert_code(self):
        tokenizer = AutoTokenizer.from_pretrained('../SemanticCodeBERT')
        bert = AutoModel.from_pretrained('../SemanticCodeBERT')
        bert.resize_token_embeddings(len(tokenizer) + 4)
        return bert

    def forward(self, Q, D, bsize=16, tag="test"):
        if tag == "test":
            q_query = self.query_q(*Q, tag)  # queries: NxC
            q_doc = self.doc_q(*D, tag)
            return self.score(q_query,q_doc)
            #return torch.einsum('nc,nc->n', [q_query, q_doc]).unsqueeze(-1)
        else:
            # compute query features
            p_query, q_query = self.query_q(*Q, "train")  # queries: NxC
            p_doc, q_doc = self.doc_q(*D, "train")

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                k_doc = self.doc_k(*D)
                k_query = self.query_k(*Q)

            halfsize = int(bsize/2)
            anchor = torch.mean(q_query, dim=1)[:halfsize,:].squeeze()
            pos = torch.mean(k_doc, dim=1)[:halfsize,:].squeeze()
            neg = torch.mean(k_doc, dim=1)[halfsize:,:].squeeze()
            l_pos = torch.einsum('nc,nc->n', [anchor, pos]).unsqueeze(-1)

            self._dequeue_and_enqueue_doc(neg)
            l_neg = torch.einsum('nc,ck->nk', [anchor, self.queue_doc.clone().detach().cuda()])

            # logits: Nx(1+K)
            logits1 = torch.cat([l_pos, l_neg], dim=1)
            # apply temperature
            logits1 /= self.T
            # labels: positive key indicators
            labels1 = torch.zeros(logits1.shape[0], dtype=torch.long).cuda()
            # dequeue and enqueue

            anchor = torch.mean(q_doc[:halfsize,:,:], dim=1).squeeze()
            pos = torch.mean(k_query[:halfsize,:,:], dim=1).squeeze()
            neg = torch.mean(k_query[halfsize:,:,:], dim=1).squeeze()
            l_pos = torch.einsum('nc,nc->n', [anchor, pos]).unsqueeze(-1)

            self._dequeue_and_enqueue_query(neg)
            l_neg = torch.einsum('nc,ck->nk', [anchor, self.queue_query.clone().detach().cuda()])

            # logits: Nx(1+K)
            logits2 = torch.cat([l_pos, l_neg], dim=1)
            # apply temperature
            logits2 /= self.T
            # labels: positive key indicators
            labels2 = torch.zeros(logits2.shape[0], dtype=torch.long).cuda()
            # dequeue and enqueue

            #self._dequeue_and_enqueue_query(pos)

            return self.score(p_query, p_doc), self.score(q_query, q_doc), logits1, labels1, logits2, labels2



    def query_q(self, input_ids, attention_mask, tag="test"):
        input_ids, attention_mask = input_ids.to(self.dev), attention_mask.to(self.dev)

        # concatenate output of the last 4 layers
        Q = self.bert_q_text(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        Q = Q[2]
        Q = torch.cat(tuple([Q[i] for i in [-4, -3, -2, -1]]), dim=-1)

        P = Q

        Q = self.linear_q_text(Q)
        Q = torch.nn.functional.normalize(Q, p=2, dim=2)
        P = torch.nn.functional.normalize(P, p=2, dim=2)

        # normalize last dimension (embeddings)
        if(tag=="train"):
            return P, Q
        return Q

    def query_k(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.dev), attention_mask.to(self.dev)

        # concatenate output of the last 4 layers
        Q = self.bert_k_text(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        Q = Q[2]
        Q = torch.cat(tuple([Q[i] for i in [-4, -3, -2, -1]]), dim=-1)

        Q = self.linear_k_text(Q)
        Q = torch.nn.functional.normalize(Q, p=2, dim=2)

        # normalize last dimension (embeddings)
        return Q

    def doc_q(self, input_ids, attention_mask, tag="test", keep_dims=True):
        if self.token_config == 'QARCD':
            context_ids = input_ids[:, : self.doc_maxlen]
            context_attention_mask = attention_mask[:, : self.doc_maxlen]
            context_ids, context_attention_mask = context_ids.to(self.dev), context_attention_mask.to(self.dev)
            Pc, context_D = self._process_doc_q(context_ids, context_attention_mask)

            added_ids = input_ids[:, self.doc_maxlen: 2 * self.doc_maxlen]
            added_attention_mask = attention_mask[:, self.doc_maxlen: 2 * self.doc_maxlen]
            added_ids, added_attention_mask = added_ids.to(self.dev), added_attention_mask.to(self.dev)
            Pa, added_D = self._process_doc_q(added_ids, added_attention_mask)

            removed_ids = input_ids[:, 2 * self.doc_maxlen:]
            removed_attention_mask = attention_mask[:, 2 * self.doc_maxlen:]
            removed_ids, removed_attention_mask = removed_ids.to(self.dev), removed_attention_mask.to(self.dev)
            Pr, removed_D = self._process_doc_q(removed_ids, removed_attention_mask)

            D = context_D + added_D - removed_D
            P = Pc + Pa - Pr

        else:
            input_ids, attention_mask = input_ids.to(self.dev), attention_mask.to(self.dev)
            P, D = self._process_doc_q(input_ids, attention_mask)

        D = torch.nn.functional.normalize(D, p=2, dim=2)
        P = torch.nn.functional.normalize(P, p=2, dim=2)

        if not keep_dims and self.token_config != 'QARCD':
            mask = torch.tensor(self.mask(input_ids), device=self.dev).unsqueeze(2).float()
            D, mask = D.to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]


        if(tag=="train"):
            return P, D
        else:
            return D

    def _process_doc_q(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.dev), attention_mask.to(self.dev)
        # D = self.bert(input_ids, attention_mask=attention_mask)[0]

        D = self.bert_q_code(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        D = D[2]
        D = torch.cat(tuple([D[i] for i in [-4, -3, -2, -1]]), dim=-1)
        P = D

        D = self.linear_q_code(D)

        mask = torch.tensor(self.mask(input_ids), device=self.dev).unsqueeze(2).float()
        D = D * mask

        return P, D
    def doc_k(self, input_ids, attention_mask, keep_dims=True):
        if self.token_config == 'QARCD':
            context_ids = input_ids[:, : self.doc_maxlen]
            context_attention_mask = attention_mask[:, : self.doc_maxlen]
            context_ids, context_attention_mask = context_ids.to(self.dev), context_attention_mask.to(self.dev)
            context_D = self._process_doc_k(context_ids, context_attention_mask)

            added_ids = input_ids[:, self.doc_maxlen: 2 * self.doc_maxlen]
            added_attention_mask = attention_mask[:, self.doc_maxlen: 2 * self.doc_maxlen]
            added_ids, added_attention_mask = added_ids.to(self.dev), added_attention_mask.to(self.dev)
            added_D = self._process_doc_k(added_ids, added_attention_mask)

            removed_ids = input_ids[:, 2 * self.doc_maxlen:]
            removed_attention_mask = attention_mask[:, 2 * self.doc_maxlen:]
            removed_ids, removed_attention_mask = removed_ids.to(self.dev), removed_attention_mask.to(self.dev)
            removed_D = self._process_doc_k(removed_ids, removed_attention_mask)

            D = context_D + added_D - removed_D

        else:
            input_ids, attention_mask = input_ids.to(self.dev), attention_mask.to(self.dev)
            D = self._process_doc_k(input_ids, attention_mask)

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims and self.token_config != 'QARCD':
            mask = torch.tensor(self.mask(input_ids), device=self.dev).unsqueeze(2).float()
            D, mask = D.to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def _process_doc_k(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.dev), attention_mask.to(self.dev)
        # D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.bert_k_code(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        D = D[2]
        D = torch.cat(tuple([D[i] for i in [-4, -3, -2, -1]]), dim=-1)
        D = self.linear_k_code(D)

        mask = torch.tensor(self.mask(input_ids), device=self.dev).unsqueeze(2).float()
        D = D * mask

        return D

    def score(self, Q, D):
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        elif self.similarity_metric == 'l2':
            return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1)) ** 2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask
