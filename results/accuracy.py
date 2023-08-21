import json
import logging
import os
import traceback
from count_exact_matches import load_br2type
from utils import get_br2skip
import argparse
import csv

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('accuracy')

granularities = ['hunks', 'commits', 'files']
types = ['fully']
techniques = ['ranking_INDEX_SemanticCodebert_RN_bertoverflow_QARCL_q256_d256_dim128_cosine_q256_d256_dim128_commits_token.tsv']


def get_details(fname):
    if not file.startswith('ranking'):
        return None

    techs = [x for x in techniques if x in fname]
    if len(techs) > 1:
        raise RuntimeError('WTF: too many techniques in {0}'.format(fname))

    grans = [x for x in granularities if x in fname]
    if len(grans) > 1:
        raise RuntimeError('WTF: too many level of granularity in {0}'.format(fname))

    if len(techs) == 0:
        return None

    if techs[0] == 'Locus':
        return 'hunks', 'Locus'

    if len(grans) == 0:
        return None

    return grans[0], techs[0]


def precision_at(ranking, goldset, k=1):
    hit = 0
    for idx, (commit, file, rank, score) in enumerate(ranking[:k]):
        commit = commit[0:7]
        if commit in goldset:
            hit += 1
        if hit == len(goldset):
            break
    return hit / min(k, len(goldset))


def ap_ranking(ranking, goldset, save=False, fpath=None, qid=None):
    ap = list()
    files = list()
    file_rank = 1
    commits_in_ranking = set()
    for idx, (commit, file, commit_rank, score) in enumerate(ranking):
        commit = commit[0:7]
        if commit in goldset:
            ap.append(file_rank / (idx + 1))
            files.append((file, idx + 1))
            file_rank += 1
            if commit in commits_in_ranking:
                raise ValueError(
                    'Should not happen. Ranking for each bug report is filtered to contain only highest scored hunk.')
            commits_in_ranking.add(commit)

    # append 0 for commits that did not make it to the ranking (faiss-depth param)
    not_found = set(goldset) - commits_in_ranking
    for _, commit in enumerate(not_found):
        ap.append(0.0)
        files.append((commit, 'N/A'))

    if save is True:
        with open(fpath, 'a') as f:
            if 'map' in fpath:
                for file, rank in files:
                    f.write('{0},{1},{2}\n'.format(qid, file, rank))
            else:
                file, rank = files[0]
                f.write('{0},{1},{2}\n'.format(qid, file, rank))

    return sum(ap) / len(ap)


def rr_ranking(ranking, goldset):
    for idx, (commit, file, rank, score) in enumerate(ranking):
        commit = commit[0:7]
        if commit in goldset:
            return 1.0 / (idx + 1)
    return 0.0


def read_goldset(fpath):
    ranks = list()
    with open(fpath) as f:
        for line in f.readlines():
            val = line.split(',')
            if len(val) > 4:
                score = val[6]
            else:
                score = val[3]
            ranks.append((int(val[0]), int(val[1]), str(val[2]), float(score.strip())))
    return ranks


def read_ranks(data_dpath, fname, save=False):
    if 'INDEX' in fname:
        granularity = fname.split('_')[-2]
    else:
        granularity = fname.split('_')[-1][:-4]
    bug2ts = load_bug2timestamp(data_dpath)
    ranking = dict()
    added_commits = dict()
    commit_rank = 1
    with open(os.path.join(data_dpath, 'results', fname)) as f:
        for line in f.readlines():
            if 'model' in fname:
                qid, pid, commit, file, score = line.strip().split('\t')
            else:
                qid, pid, commit, file, rank, score = line.strip().split('\t')

            qid = int(qid)

            if qid not in ranking:
                ranking[qid] = list()
                added_commits[qid] = set()
                commit_rank = 1

            if commit in added_commits[qid]:
                continue

            if 'Locus' not in fname:
                with open(os.path.join(data_dpath, granularity, file)) as f:
                    timestamp = float(json.load(f)['timestamp'])

                bug_ts = bug2ts[qid]
                if timestamp > bug_ts:
                    continue

            ranking[qid].append((commit, file, commit_rank, score))
            added_commits[qid].add(commit)
            commit_rank += 1

    if save is True:
        fpath = os.path.join(data_dpath, 'results', 'filtered_' + fname)
        with open(fpath, 'w') as f:
            for qid in ranking:
                for commit, file, commit_rank, score in ranking[qid]:
                    f.write('\t'.join([str(qid), commit, file, str(commit_rank), str(score)]) + '\n')
    return ranking


def load_bug2timestamp(data_dpath):
    bug2ts = dict()
    with open(os.path.join(data_dpath, 'open_ts.txt')) as f:
        for line in f.readlines():
            bid, ts = line.strip().split(',')
            bid = int(bid)
            ts = float(ts)
            if bid in bug2ts:
                raise RuntimeError('This should not have happened.')
            bug2ts[bid] = ts
    return bug2ts


def load_issue2git(file):
    issue2git = dict()
    with open(file) as f:
        f.readline()
        for line in f.readlines():
            qid, _, sha = line.strip().split('\t')
            qid = int(qid)
            if qid not in issue2git:
                issue2git[qid] = list()
            issue2git[qid].append(sha[0:7])
    return issue2git


def metrics(data_dpath, ranks_file, br_to_skip, br2type, use_types):
    ranks = read_ranks(data_dpath, ranks_file)
    issue2git = load_issue2git(os.path.join(data_dpath, 'issue2git.tsv'))
    aps = list()
    rrs = list()
    p1 = list()
    p3 = list()
    p5 = list()

    sorted_qid = sorted(ranks.keys())
    for qid in sorted_qid:
        ranking = ranks[qid]
        if qid not in issue2git:
            continue

        # skip BR which does not belong to train or test set
        if br_to_skip is not None and qid in br_to_skip:
            continue

        # skip BR which does not belong to given type
        if br2type[str(qid)] not in use_types:
            continue

        goldset = issue2git[qid]

        # MAP
        ap = ap_ranking(ranking, goldset)
        aps.append(ap)

        # MRR
        rr = rr_ranking(ranking, goldset)
        rrs.append(rr)

        # Precision@K
        p1.append(precision_at(ranking, goldset, 1))
        p3.append(precision_at(ranking, goldset, 3))
        p5.append(precision_at(ranking, goldset, 5))

    mrr = sum(rrs) / len(rrs)
    map = sum(aps) / len(aps)
    p1_avg = sum(p1) / len(p1)
    p3_avg = sum(p3) / len(p3)
    p5_avg = sum(p5) / len(p5)
    count = len(rrs)
    logger.info('CONFIG: {0:<70}\t{1:<5}\t{2:<5}\t{3:<5}\t{4:<5}\t{5:<5}\t{6:<3}'.format(data_dpath + '/' + ranks_file,
                                                                                         round(mrr, 3), round(map, 3),
                                                                                         round(p1_avg, 3),
                                                                                         round(p3_avg, 3),
                                                                                         round(p5_avg, 3), count))
    return rrs, aps, p1, p3, p5


def update_results(fpath, results, eval_type, technique, project, gran, rrs, aps, p1, p3, p5):
    logger.info('Record results for {0} - {1} - {2} - {3}'.format(eval_type, technique, project, gran))
    if technique not in results[eval_type]:
        results[eval_type][technique] = dict()

    if project not in results[eval_type][technique]:
        results[eval_type][technique][project] = dict()

    if gran not in results[eval_type][technique][project]:
        results[eval_type][technique][project][gran] = dict()
    elif 'MRR' in results[eval_type][technique][project][gran]:
        raise ValueError(
            'Found more than one file with results for {0} - {1} - {2}. Current file {3}'.format(
                technique, project_name, gran, file))

    results[eval_type][technique][project][gran]['MRR'] = rrs
    results[eval_type][technique][project][gran]['MAP'] = aps
    results[eval_type][technique][project][gran]['P@1'] = p1
    results[eval_type][technique][project][gran]['P@3'] = p3
    results[eval_type][technique][project][gran]['P@5'] = p5
    results[eval_type][technique][project][gran]['fpath'] = fpath

    return results


def compute_metrics(results):
    results2save = list()
    for eval_type in results:
        for technique in results[eval_type]:
            technique_results = {}
            for project in results[eval_type][technique]:
                logger.info('Saving results for project {0}'.format(project))
                for gran in results[eval_type][technique][project]:
                    metrics = results[eval_type][technique][project][gran]
                    MRR = sum(metrics['MRR']) / len(metrics['MRR'])
                    MAP = sum(metrics['MAP']) / len(metrics['MAP'])
                    P1 = sum(metrics['P@1']) / len(metrics['P@1'])
                    P3 = sum(metrics['P@3']) / len(metrics['P@3'])
                    P5 = sum(metrics['P@5']) / len(metrics['P@5'])
                    results2save.append(
                        (eval_type, technique, project, gran, MRR, MAP, P1, P3, P5,
                         len(metrics['MRR']), metrics['fpath']))

                    if gran not in technique_results:
                        technique_results[gran] = {'MRR': list(), 'MAP': list(), 'P@1': list(), 'P@3': list(),
                                                   'P@5': list(), 'fpaths': ''}
                    technique_results[gran]['MRR'].extend(metrics['MRR'])
                    technique_results[gran]['MAP'].extend(metrics['MAP'])
                    technique_results[gran]['P@1'].extend(metrics['P@1'])
                    technique_results[gran]['P@3'].extend(metrics['P@3'])
                    technique_results[gran]['P@5'].extend(metrics['P@5'])
                    technique_results[gran]['fpaths'] += '; ' + metrics['fpath']

            # summary per technique and granularity across all projects
            for gran in technique_results:
                metrics = technique_results[gran]
                MRR = sum(metrics['MRR']) / len(metrics['MRR'])
                MAP = sum(metrics['MAP']) / len(metrics['MAP'])
                P1 = sum(metrics['P@1']) / len(metrics['P@1'])
                P3 = sum(metrics['P@3']) / len(metrics['P@3'])
                P5 = sum(metrics['P@5']) / len(metrics['P@5'])
                results2save.append(
                    (eval_type, technique, 'all', gran, MRR, MAP, P1, P3, P5,
                     len(metrics['P@5']), metrics['fpaths']))

    return results2save


def save_results(results, args):

    for values in results:
        with open("retrieval_result.csv", "a+", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([str(x) for x in values] + [fpath, args.n_epochs, args.bsize, args.k, args.dataset])  # 先写入columns_name



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-epochs', type=int, default=4)
    parser.add_argument('--bsize',default=8, type=int)
    parser.add_argument('--k', default=128, type=int)
    parser.add_argument('--dataset', default="zxing")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    results_dpath = '/home/sdu/dyl/bug/data/'
    evaluate_on = 'test'
    project_name = args.dataset
    results = {'fully': dict(), 'partially': dict(), 'not_localized': dict(), 'not_localized+partially': dict(),
              'partially+not_localized+fully': dict()}

    
    try:
        for eval_type in types:
            #for project_name in projects:
                project_dpath = os.path.join(results_dpath, project_name)
                br2type = load_br2type('/home/sdu/dyl/bug/data/' + project_name)
                br_to_skip = get_br2skip(project_dpath, evaluate_on)

                project_results_dpath = os.path.join(project_dpath, 'results')
                for file in os.listdir(project_results_dpath):
                    details = get_details(file)

                    if details is None:
                        logger.info('Skipping {0}. Unknown details'.format(project_name + '/' + file))
                        continue

                    gran, technique = details
                    fpath = os.path.join(project_results_dpath, file)

                    rrs, aps, p1, p3, p5 = metrics(project_dpath, file, br_to_skip, br2type, eval_type)

                    results = update_results(fpath, results, eval_type, technique, project_name, gran, rrs, aps, p1, p3, p5)


    except Exception as e:
        logger.error(traceback.format_exc())

    results = compute_metrics(results)
    save_results(results, args)
