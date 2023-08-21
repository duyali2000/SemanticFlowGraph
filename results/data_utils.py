import io
import json
import logging
import os
import re

import dulwich.client
import dulwich.repo
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', )
logger = logging.getLogger('data-utils')


def _clone(source, target, bare=False):
    client, host_path = dulwich.client.get_transport_and_path(source)

    if target is None:
        target = host_path.split("/")[-1]

    if not os.path.exists(target):
        os.mkdir(target)

    if bare:
        r = dulwich.repo.Repo.init_bare(target)
    else:
        r = dulwich.repo.Repo.init(target)

    remote_refs = client.fetch(host_path, r, determine_wants=r.object_store.determine_wants_all)

    r["HEAD".encode('utf-8')] = remote_refs["HEAD".encode('utf-8')]

    for key, val in remote_refs.items():
        if not key.endswith('^{}'.encode('utf-8')):
            r.refs.add_if_new(key, val)

    return r


def _load(repo_url):
    repos_base = 'gits'
    if not os.path.exists(repos_base):
        os.makedirs(repos_base)

    repo_name = repo_url.split('/')[-1]
    target = os.path.join(repos_base, repo_name)
    try:
        repo = _clone(repo_url, target, bare=True)
    except OSError:
        repo = dulwich.repo.Repo(target)

    return repo


def load_repo(url, ref='HEAD'):
    repo = _load(url)

    ref_tree = None
    ref_commit_sha = None
    try:
        ref_obj = repo[ref.encode('utf-8')]
    except:
        logger.info('Could not find ref %s in repo, using HEAD', ref)
        ref_obj = repo[repo.head()]

    if isinstance(ref_obj, dulwich.objects.Tag):
        ref_tree = repo[ref_obj.object[1]].tree
        ref_commit_sha = ref_obj.object[1]
    elif isinstance(ref_obj, dulwich.objects.Commit):
        ref_tree = ref_obj.tree
        ref_commit_sha = ref_obj.id
    elif isinstance(ref_obj, dulwich.objects.Tree):
        ref_tree = ref_obj.id
    else:
        ref_tree = ref
    return repo, ref_commit_sha


def _get_diff(repo, changeset):
    patch_file = io.BytesIO()
    try:
        dulwich.patch.write_object_diff(patch_file,
                                        repo.object_store,
                                        changeset.old, changeset.new)
    except UnicodeDecodeError as e:
        logger.debug(e)
        return ''

    return patch_file.getvalue()


def _walk_changes(repo, ref_commit_sha):
    for walk_entry in repo.get_walker(include=[ref_commit_sha], reverse=True):
        commit = walk_entry.commit

        # initial revision, has no parent
        if len(commit.parents) == 0:
            for changes in dulwich.diff_tree.tree_changes(
                    repo.object_store, None, commit.tree):
                diff = _get_diff(repo, changes)
                yield commit, None, diff

        for parent in commit.parents:
            # do I need to know the parent id?

            for changes in dulwich.diff_tree.tree_changes(
                    repo.object_store, repo[parent].tree, commit.tree):
                diff = _get_diff(repo, changes)
                yield commit, parent, diff


def get_commits(repo, ref_commit_sha, include_additions=True, include_removals=True, include_context=True,
                divide_commits=False):
    if divide_commits is True:
        unified = re.compile(r'^[+ -@].*')
    else:
        unified = re.compile(r'^[+ -].*')
    context = re.compile(r'^ .*')
    addition = re.compile(r'^\+.*')
    removal = re.compile(r'^-.*')
    current = None
    msg = None
    date = None
    metainfo = None
    low = ''  # collecting the list of words

    for commit, parent, diff in _walk_changes(repo, ref_commit_sha):
        # write out once all diff lines for commit have been collected
        # this is over all parents and all files of the commit
        diff = diff.decode("utf-8", errors='ignore')
        if current is None:
            # set current for the first commit, clear low
            current = commit.id.decode("utf-8")
            msg = commit.message.decode('utf-8')
            date = commit.commit_time
            low = ''
        elif current != commit.id.decode("utf-8") and not divide_commits:
            # new commit seen, yield the collected low
            yield low, msg, current, date, ''

            current = commit.id.decode("utf-8", errors='ignore')
            msg = commit.message.decode('utf-8', errors='ignore')
            date = commit.commit_time
            low = ''
        elif divide_commits:
            current = commit.id.decode("utf-8")

        diff_lines = list(filter(lambda x: unified.match(x), diff.splitlines()))

        if len(diff_lines) < 2:
            continue  # useful for not worrying with binary files

        # discard non *.java files
        if re.compile(r'.*\.java').match(diff_lines[1]) is None and re.compile(r'.*\.kt').match(
                diff_lines[1]) is None:
            continue

        lines = diff_lines

        if not include_additions:
            lines = filter(lambda x: not addition.match(x), lines)

        if not include_removals:
            lines = filter(lambda x: not removal.match(x), lines)

        if not include_context:
            lines = filter(lambda x: not context.match(x), lines)

        # keep unified markers, so we can mark added/removed/context lines
        # with special BERT tokens
        # lines = [line[1:] for line in lines]  # remove unified markers
        msg = commit.message.decode("utf-8", errors='ignore')
        msg = preprocess_msg(msg)
        date = commit.commit_time

        document = '\n'.join(lines)

        if divide_commits is True:
            header = lines[1]
            doc = header
            metainfo = header + lines[2]
            for line in lines[3:]:
                if line.startswith('@ ') or line.startswith('@@ '):
                    doc = remove_license(doc)
                    yield doc, msg, current, date, metainfo
                    doc = header
                    metainfo = header + ' ' + line
                else:
                    doc += '\n' + line

            doc = remove_license(doc)
            yield doc, msg, current, date, metainfo

        low += '\n' + document

    if not divide_commits:
        yield low, msg, current, date, ''


def preprocess_msg(msg):
    msg = msg.strip().replace('\n', ' ')
    # cut off git-svn-id
    if msg.find('git-svn-id:') != -1:
        msg = msg[: msg.find('git-svn-id:')]
    return msg


def remove_license(hunk):
    block_comment = False
    license_comment = False
    lines_to_remove = set()
    lines = hunk.split('\n')
    for idx, line in enumerate(lines):
        line_no_markers = line[1:].strip().lower()
        if line_no_markers.startswith('/*'):
            block_comment = True
            lines_to_remove.add(idx)
        elif line_no_markers.startswith('*/'):
            lines_to_remove.add(idx)
            break
        elif line_no_markers.startswith('*') and block_comment == True:
            if 'license' in line_no_markers or 'copyright' in line_no_markers:
                license_comment = True
            lines_to_remove.add(idx)

    if license_comment:
        hunk = '\n'.join([line for idx, line in enumerate(lines) if idx not in lines_to_remove])

    return hunk


def load_issue2fixing(project_dpath):
    df = pd.read_csv(os.path.join(project_dpath, 'issue2fix.tsv'), sep='\t')
    issue2git = dict()
    for idx, row in df.iterrows():
        bid = int(row['issue'])
        sha = row['sha']

        if bid not in issue2git:
            issue2git[bid] = list()
        issue2git[bid].append(sha)

    return issue2git


def load_goldset(project_dpath, by_ids=[]):
    df = pd.read_csv(os.path.join(project_dpath, 'issue2git.tsv'), sep='\t')
    issue2git = dict()
    git2issue = dict()
    for idx, row in df.iterrows():
        bid = int(row['issue'])
        sha = row['sha']

        if len(by_ids) > 0 and bid not in by_ids:
            continue

        if bid not in issue2git:
            issue2git[bid] = list()
        issue2git[bid].append(sha)

        if sha not in git2issue:
            git2issue[sha] = list()
        git2issue[sha].append(bid)

    return issue2git, git2issue


def load_commits(project_path, commit_limit=None, include_log=False, sha2include=[], short_sha=False):
    commit_list = list()
    sha_list = list()
    hunks = list()
    timestamps = list()
    commits_dpath = os.path.join(project_path, 'commits')

    if commit_limit is None or commit_limit > 0:
        for file in os.listdir(commits_dpath):
            if not file.startswith('c_'):
                continue

            with open(os.path.join(commits_dpath, file)) as f:
                data = json.load(f)
                commit = data['commit']
                sha = data['sha']
                hunk = data['metainfo']
                ts = data['timestamp']

                if include_log is True:
                    commit = data['log'] + ' ' + commit

            if len(commit) == 0:
                continue

            commit_list.append(commit)
            hunks.append(hunk)
            timestamps.append(ts)
            if short_sha:
                sha_list.append(sha[0:7])
            else:
                sha_list.append(sha)

            if len(commit_list) % 1000 == 0:
                print('Hunks processed {0}'.format(len(commit_list)))

            if commit_limit is not None and len(commit_list) > commit_limit:
                break

    added_sha = set(sha_list)
    for sha in sha2include:
        if sha in added_sha:
            continue
        for file in os.listdir(commits_dpath):
            if short_sha is True:
                commit = file[2:9]
            else:
                commit = file
            if sha in commit:
                with open(os.path.join(commits_dpath, file)) as f:
                    data = json.load(f)
                    commit = data['commit']
                    if include_log is True:
                        commit = data['log'] + ' ' + commit

                    commit_list.append(commit)
                    hunks.append(data['metainfo'])
                    timestamps.append(data['timestamp'])

                    if short_sha:
                        sha_list.append(data['sha'][0:7])
                    else:
                        sha_list.append(data['sha'])

    return commit_list, sha_list, hunks, timestamps


def load_brs(project_path, br_no=[]):
    queries = list()
    br_dpath = os.path.join(project_path, 'br')
    shortq_dir = os.path.join(br_dpath, 'short')
    longq_dir = os.path.join(br_dpath, 'long')

    for file in os.listdir(shortq_dir):
        if not file.endswith('.txt'):
            continue

        with open(os.path.join(shortq_dir, file), errors='ignore') as f:
            query = '\n'.join(f.readlines())

        with open(os.path.join(longq_dir, file), errors='ignore') as f:
            query += '\n ' + '\n'.join(f.readlines())

        queries.append((int(file[:-4]), query))

    if len(br_no) > 0:
        filtered_queries = list()
        for bid, query in queries:
            if bid in br_no:
                filtered_queries.append((int(bid), query))
        queries = filtered_queries

    bug_ids = [bug[0] for bug in queries]
    text = [bug[1] for bug in queries]
    br2ts = load_br2timestamps(project_path)

    return bug_ids, text, br2ts


def load_br2timestamps(project_path, ts_type='open_ts'):
    br2ts = dict()
    with open(os.path.join(project_path, ts_type + '.txt')) as f:
        for line in f.readlines():
            bid, ts = line.strip().split(',')
            if int(bid) in br2ts:
                raise RuntimeError('This should not happen.')
            br2ts[int(bid)] = float(ts)
    return br2ts


def valid_commit(timestamp, limit_ts):
    # remove all commits that occurred after the last bug in the test set was closed
    if float(timestamp) > limit_ts:
        return False
    return True


def get_limit_ts(dpath):
    fixed_ts = list()
    with open(os.path.join(dpath, 'fix_ts.txt')) as f:
        for line in f.readlines():
            fixed_ts.append(float(line.split(',')[1].strip()))

    # sort fixing times and get the newest timestamp to make sure all bugs are closed by this time
    # which means that all introducing commits must have appeared
    return sorted(fixed_ts, reverse=True)[0]


def collect_commits(repo_url, project_name, ref=None, divide_commits=True):
    logger.info('Processing {0}'.format(project_name))

    project_dpath = os.path.join('../data', project_name)
    if divide_commits is True:
        output_dpath = os.path.join(project_dpath, 'hunks/')
    else:
        output_dpath = os.path.join(project_dpath, 'commits/')
    limit_ts = get_limit_ts(project_dpath)
    if not os.path.exists(output_dpath):
        os.makedirs(output_dpath)

    repo, ref_sha = load_repo(repo_url, ref)
    sha_cnt = dict()

    for commit, log, sha, date, metainfo in get_commits(repo, ref_sha, divide_commits=divide_commits):
        if sha not in sha_cnt:
            sha_cnt[sha] = 0
        else:
            sha_cnt[sha] = sha_cnt[sha] + 1

        cnt = sha_cnt[sha]
        if divide_commits is False and cnt == 1:
            raise RuntimeError('More than one commit with the same sha? Something is wrong')

        if valid_commit(date, limit_ts):
            with open(os.path.join(output_dpath, 'c_{0}_{1}.json'.format(sha, cnt)), 'w') as f:
                json.dump({'sha': sha, 'log': log, 'commit': commit, 'timestamp': date, 'metainfo': metainfo}, f)


if __name__ == '__main__':

    repo_urls = [('git://github.com/eclipse/eclipse.platform.swt.git', 'swt')]
    # ('git://github.com/apache/tomcat.git', 'tomcat'),
    # ('git://github.com/zxing/zxing.git', 'zxing'),
    # ('git://github.com/eclipse/eclipse.jdt.core.git', 'jdt'),
    # ('git://github.com/eclipse/eclipse.pde.ui.git', 'pde'),
    # ('git://github.com/eclipse/org.aspectj.git', 'aspectj')]

    for url, project_name in repo_urls:
        logger.info('Processing {0}'.format(url))
        collect_commits(url, project_name)
