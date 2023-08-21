import json
import os
import re

from data_utils import load_goldset, load_brs, load_issue2fixing


def get_classnames(data_dir):
    data_dir = os.path.join(data_dir, 'hunks')
    classnames = set()
    for fname in os.listdir(data_dir):
        if not fname.startswith('c_'):
            continue

        with open(os.path.join(data_dir, fname)) as f:
            data = json.load(f)

        cname = data['commit'].split('\n')[0][3:].split('/')[-1].split('.')[0].strip()
        classnames.add(cname)

    return classnames


def get_introducing_classnames(data_dir):
    issue2git, _ = load_goldset(data_dir)
    issues2class = dict()
    for bid in issue2git:
        classes = set()
        shas = set(issue2git[bid])

        for fname in os.listdir(os.path.join(data_dir, 'files')):
            sha_fname = fname.split('_')[1][0:7]
            if sha_fname in shas:
                with open(os.path.join(data_dir, 'files', fname)) as f:
                    data = json.load(f)

                fname = data['commit'].split('\n')[0].strip()[3:].split('/')[-1].strip()

                if not fname.endswith('.java'):
                    continue

                cname = fname.split('.')[0].strip()

                if 'test' in cname or 'Test' in cname:
                    continue

                classes.add(cname)

        issues2class[int(bid)] = classes

    return issues2class


def get_fixing_classnames(data_dir):
    issue2fix = load_issue2fixing(data_dir)
    issues2class = dict()
    for bid in issue2fix:
        classes = set()
        shas = set(issue2fix[bid])

        for fname in os.listdir(os.path.join(data_dir, 'files')):
            sha_fname = fname.split('_')[1]
            if sha_fname in shas:
                with open(os.path.join(data_dir, 'files', fname)) as f:
                    data = json.load(f)

                fname = data['commit'].split('\n')[0].strip()[3:].split('/')[-1].strip()

                if not fname.endswith('.java'):
                    continue

                cname = fname.split('.')[0].strip()

                if 'test' in cname or 'Test' in cname:
                    continue

                classes.add(cname)

        issues2class[int(bid)] = classes

    return issues2class


def load_br2type(data_dir):
    fpath = os.path.join(data_dir, 'br_types.json')

    if os.path.exists(fpath):
        with open(fpath) as f:
            br2type = json.load(f)
        return br2type

    br2type = dict()
    bug_ids, text, br2ts = load_brs(data_dir)
    issue2classname = get_fixing_classnames(data_dir)

    not_loc = 0
    partially = 0
    fully = 0

    for bid, query in list(zip(bug_ids, text)):
        if bid not in issue2classname:
            continue
        hints = set()
        tokens = re.split(' |\.|/|;|#|>|\$|\(|\[|\-|=|\+|\|@|', query)

        for token in tokens:
            if token in issue2classname[bid]:
                hints.add(token)

        goldset = issue2classname[bid]
        if len(goldset) == 0:
            print(bid)
            br2type[str(bid)] = 'remove'
        elif len(hints) == 0:
            br2type[str(bid)] = 'not_localized'
            not_loc += 1
        else:
            difference = issue2classname[int(bid)] - hints
            if len(difference) == 0:
                br2type[str(bid)] = 'fully'
                fully += 1
            else:
                br2type[str(bid)] = 'partially'
                partially += 1

    with open(fpath, 'w') as f:
        json.dump(br2type, f)

    return br2type
