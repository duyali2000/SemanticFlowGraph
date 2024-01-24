import torch

from transformers import BertTokenizerFast, RobertaTokenizer, AutoTokenizer, AutoModel


class DocTokenizer:
    def __init__(self, config, doc_maxlen, special_tokens_cfg):
        self.doc_maxlen = doc_maxlen
        self.special_tokens_cfg = special_tokens_cfg
        self.tok, self.special_tokens = self._create_tokenizer(config)

        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id

    def _create_tokenizer(self, config):
        tok = load_tokenizer_doc(config)

        special_tokens = dict()
        if self.special_tokens_cfg == 'QARCL' or self.special_tokens_cfg == 'QARC' or self.special_tokens_cfg == 'QARCD':
            if config == 'bert-base-uncased':
                tok.add_special_tokens({'additional_special_tokens': ['[unused5]', '[unused6]', '[unused7]','[unused8]']})
                special_tokens['C'] = '[unused5]'
                special_tokens['A'] = '[unused6]'
                special_tokens['R'] = '[unused7]'
                special_tokens['V'] = '[unused8]'
            else:
                tok.add_special_tokens({'additional_special_tokens': ['[UNUSED_5]', '[UNUSED_6]', '[UNUSED_7]','[UNUSED_8]']})
                special_tokens['C'] = '[UNUSED_5]'
                special_tokens['A'] = '[UNUSED_6]'
                special_tokens['R'] = '[UNUSED_7]'
                special_tokens['V'] = '[UNUSED_8]'

        else:
            if config == 'bert-base-uncased':
                tok.add_special_tokens({'additional_special_tokens': ['[unused5]','[unused6]']})
                special_tokens['D'] = '[unused5]'
                special_tokens['V'] = '[unused6]'
            else:
                tok.add_special_tokens({'additional_special_tokens': ['[UNUSED_5]','[UNUSED_6]']})
                special_tokens['D'] = '[UNUSED_5]'
                special_tokens['V'] = '[UNUSED_6]'

        return tok, special_tokens

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))
        #assert type(batch_log) in [list, tuple], (type(batch_log))

        # divide hunk based on special token option
        if self.special_tokens_cfg == 'QARCL':
            batch_text = [
                divide_hunk_lines(x, context_tok=self.special_tokens['C'], added_tok=self.special_tokens['A'],
                                  removed_tok=self.special_tokens['R'], nodesequence_tok=self.special_tokens['V']) for x in batch_text]
            batch_text = [x.replace(';', ' [SEP] ') for x in batch_text]
        elif self.special_tokens_cfg == 'QARC':
            batch_text = [
                divide_hunk(x, context_tok=self.special_tokens['C'], added_tok=self.special_tokens['A'],
                            removed_tok=self.special_tokens['R'], nodesequence_tok=self.special_tokens['V'],concate=True) for x in batch_text]
            batch_text = [x.replace(';', ' [SEP] ') for x in batch_text]

        elif self.special_tokens_cfg == 'QARCD':
            batch_text = [
                divide_hunk(x, context_tok=self.special_tokens['C'], added_tok=self.special_tokens['A'],
                            removed_tok=self.special_tokens['R'], nodesequence_tok=self.special_tokens['V'], concate=False) for x in batch_text]

            batch_context = [x[0] for x in batch_text]
            batch_context = [x.replace(';', ' [SEP] ') for x in batch_context]
            batch_added = [x[1] for x in batch_text]
            batch_added = [x.replace(';', ' [SEP] ') for x in batch_added]
            batch_removed = [x[2] for x in batch_text]
            batch_removed = [x.replace(';', ' [SEP] ') for x in batch_removed]

        else:
            batch_text = [self.special_tokens['D'] + ' ' + process_hunk(x, nodesequence_tok=self.special_tokens['V']) for x in batch_text]
            batch_text = [x.replace(';', ' [SEP] ') for x in batch_text]


        # using __call__ on tokenizer encodes text with add_special_tokens=True by default
        if self.special_tokens_cfg != 'QARCD':
            obj = self.tok(batch_text, padding='longest', truncation='longest_first', return_tensors='pt',
                           max_length=self.doc_maxlen)
            ids, mask = obj['input_ids'], obj['attention_mask']
        else:
            obj_c = self.tok(batch_context, padding='max_length', truncation='longest_first', return_tensors='pt',
                             max_length=self.doc_maxlen)
            obj_a = self.tok(batch_added, padding='max_length', truncation='longest_first', return_tensors='pt',
                             max_length=self.doc_maxlen)
            obj_r = self.tok(batch_removed, padding='max_length', truncation='longest_first', return_tensors='pt',
                             max_length=self.doc_maxlen)
            ids = torch.cat((obj_c['input_ids'], obj_a['input_ids'], obj_r['input_ids']), 1)
            mask = torch.cat((obj_c['attention_mask'], obj_a['attention_mask'], obj_r['attention_mask']), 1)

        if bsize:
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices

        return ids, mask


class QueryTokenizer:

    def __init__(self, config, query_maxlen):
        self.tok = load_tokenizer(config)
        self.query_maxlen = query_maxlen
        if config == 'bert-base-uncased':
            self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tok.get_vocab()['[unused0]']
            self.tok.add_special_tokens({'additional_special_tokens': ['[unused0]']})
        else:
            self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tok.get_vocab()['[UNUSED_0]']
            self.tok.add_special_tokens({'additional_special_tokens': ['[UNUSED_0]']})

        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id
        self.padding_token, self.padding_token_id = self.tok.pad_token, self.tok.pad_token_id

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix + [self.mask_token] * (self.query_maxlen - (len(lst) + 3)) for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix + [self.mask_token_id] * (self.query_maxlen - (len(lst) + 3)) for lst in ids]

        return ids

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [Q] marker
        batch_text = ['. ' + x for x in batch_text]

        obj = self.tok(batch_text, padding='max_length', truncation=True,
                       return_tensors='pt', max_length=self.query_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == self.padding_token_id] = self.mask_token_id

        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches

        return ids, mask


def tensorize_triples(query_tokenizer, doc_tokenizer, queries, positive_hunk, negative_hunk, bsize):
    assert bsize is None or len(queries) % bsize == 0

    N = len(queries)
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    D_ids, D_mask = doc_tokenizer.tensorize(positive_hunk + negative_hunk)
    D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask

    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)

    batches = []
    for (q_ids, q_mask), (p_ids, p_mask), (n_ids, n_mask) in zip(query_batches, positive_batches, negative_batches):
        Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
        D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))
        batches.append((Q, D))
    return batches


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset + bsize], mask[offset:offset + bsize]))

    return batches


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def load_tokenizer(config):
    if 'codebert' in config:
        tok = RobertaTokenizer.from_pretrained(config)
        tok.add_tokens(['[UNUSED_0]', '[UNUSED_1]', '[UNUSED_2]', '[UNUSED_3]'], special_tokens=True)
    else:
        tok = BertTokenizerFast.from_pretrained(config)
    return tok


def load_tokenizer_doc(config):
    tok = AutoTokenizer.from_pretrained('../SemanticCodeBERT')
    return tok

def divide_hunk(hunk, context_tok='[UNUSED_1]', added_tok='[UNUSED_2]', removed_tok='[UNUSED_3]', nodesequence_tok='[UNUSED_4]', concate=True):
    context = []
    added = []
    removed = []

    contextvs = []
    addedvs = []
    removedvs = []

    for line in hunk.split('\n'):
        line = line.strip()
        if len(line) < 2:
            continue
        if line.startswith('+++'):
            context.append(line[3:].strip())

        elif line.startswith('+'):
            added.append(line[1:].strip())

        elif line.startswith('-'):
            removed.append(line[1:].strip())

        else:
            context.append(line)

    if concate is True:
        return context_tok + ' ' + ' '.join(context) + \
               added_tok + ' ' + ' '.join(added) + \
               removed_tok + ' ' + ' '.join(removed) + \
               nodesequence_tok + ' ' + ' '.join(contextvs) + \
               nodesequence_tok + ' ' + ' '.join(addedvs) + \
               nodesequence_tok + ' ' + ' '.join(removedvs)

    else:
        return [context_tok + ' ' + ' '.join(context) + nodesequence_tok + ' ' + ' '.join(contextvs),
                added_tok + ' ' + ' '.join(added) + nodesequence_tok + ' ' + ' '.join(addedvs),
                removed_tok + ' ' + ' '.join(removed) + nodesequence_tok + ' ' + ' '.join(removedvs)
                ]


def divide_hunk_lines(hunk, context_tok='[UNUSED_1]', added_tok='[UNUSED_2]', removed_tok='[UNUSED_3]', nodesequence_tok='[UNUSED_4]'):
    current_token = context_tok
    lines = list()
    lines.append(current_token)
    for line in hunk.split('\n'):
        line = line.strip()
        if len(line) < 2:
            continue
        if line.startswith('+++'):
            current_token = context_tok
            lines.append(line[3:])
        elif line.startswith('+'):
            if current_token != added_tok:
                current_token = added_tok
                lines.append(added_tok)
            lines.append(line[1:].strip())
        elif line.startswith('-'):
            if current_token != removed_tok:
                current_token = removed_tok
                lines.append(removed_tok)
            lines.append(line[1:].strip())
        else:
            if current_token != context_tok:
                current_token = context_tok
                lines.append(context_tok)
            lines.append(line)
    current_token = context_tok
    for line in hunk.split('\n'):
        line = line.strip()
        if line.startswith('+++'):
            current_token = context_tok
        elif line.startswith('+'):
            if current_token != added_tok:
                current_token = added_tok
                lines.append(nodesequence_tok)
        elif line.startswith('-'):
            if current_token != removed_tok:
                current_token = removed_tok
                lines.append(nodesequence_tok)
        else:
            if current_token != context_tok:
                current_token = context_tok
                lines.append(nodesequence_tok)
    return ' '.join(lines)


def process_hunk(hunk, nodesequence_tok='[UNUSED_4]'):
    lines = []
    vs = []
    for line in hunk.split('\n'):
        line = line.strip()
        if len(line) < 2:
            continue
        if line.startswith('+++') or line.startswith('---'):
            lines.append(line[3:])
        elif line.startswith('+') or line.startswith('-'):
            lines.append(line[1:].strip())
        else:
            lines.append(line)
    for line in hunk.split('\n'):
        line = line.strip()
        if len(line) < 2:
            continue
        if line.startswith('+++') or line.startswith('---'):
            a_set = set(line[3:].split(' '))
            vs.extend(a_set)
        elif line.startswith('+') or line.startswith('-'):
            a_set = set(line[1:].split(' '))
            vs.extend(a_set)
        else:
            a_set = set(line.split(' '))
            vs.extend(a_set)
    return ' '.join(lines) + nodesequence_tok + ' '.join(vs)

