import torch

from SemanticCodebert import SemanticCodebert
from manager import MixedPrecisionManager
from tokenizer import QueryTokenizer, DocTokenizer
from utils_colbert import get_special_tokens


class ModelInference:
    def __init__(self, colbert: SemanticCodebert, args, amp=False):
        assert colbert.training is False

        self.special_tokens = get_special_tokens(args.checkpoint)
        self.emb_cmp = args.embeddings_comparison
        self.colbert = colbert
        self.query_tokenizer = QueryTokenizer(self.colbert.config, args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(self.colbert.config, args.doc_maxlen, self.special_tokens)

        self.amp_manager = MixedPrecisionManager(amp)

    def query(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                Q = self.colbert.query_q(*args, **kw_args)
                if 'average' in self.emb_cmp:
                    # unsqueeze(0) to be compatible with ColBERT code;
                    # ColBERT needs to get (#words, #dim) per document,
                    # so unsqueeze(0) will make it (1, #dim) which is fine
                    Q = torch.mean(Q, dim=1).unsqueeze(0)
                return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                D = self.colbert.doc_q(*args, **kw_args)
                if 'average' in self.emb_cmp:
                    # unsqueeze(0) to be compatible with ColBERT code;
                    # ColBERT needs to get (#words, #dim) per document,
                    # so unsqueeze(0) will make it (1, #dim) which is fine
                    D = [torch.mean(d, dim=0).unsqueeze(0) for d in D]
                return D.cpu() if to_cpu else D

    def queryFromText(self, queries, bsize=None, to_cpu=False):
        if bsize:
            batches = self.query_tokenizer.tensorize(queries, bsize=bsize)
            batches = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in batches]
            return torch.cat(batches)

        input_ids, attention_mask = self.query_tokenizer.tensorize(queries)
        Q = self.query(input_ids, attention_mask)
        return Q, attention_mask

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False):
        if bsize:
            batches, reverse_indices = self.doc_tokenizer.tensorize(docs, bsize=bsize)

            batches = [self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)
                       for input_ids, attention_mask in batches]

            if keep_dims:
                D = _stack_3D_tensors(batches)
                return D[reverse_indices]

            D = [d for batch in batches for d in batch]
            return [D[idx] for idx in reverse_indices.tolist()]

        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims)

    def score(self, Q, D, mask=None, lengths=None):
        if lengths is not None:
            assert mask is None, "don't supply both mask and lengths"

            mask = torch.arange(D.size(1), device=self.colbert.dev) + 1
            mask = mask.unsqueeze(0) <= lengths.to(self.colbert.dev).unsqueeze(-1)

        scores = (D @ Q)
        scores = scores if mask is None else scores * mask.unsqueeze(-1)
        scores = scores.max(1)

        return scores.values.sum(-1).cpu()


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos

    return output
