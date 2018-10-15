import os

import pandas as pd


def _file_paths(base_dir, task):
    return {
        'E.c': {
            'train': os.path.join(base_dir, 'E-c-En', 'train', '2018-E-c-En-train.txt'),
            'dev': os.path.join(base_dir, 'E-c-En', 'dev', '2018-E-c-En-dev.txt'),
            'test': os.path.join(base_dir, 'E-c-En', 'test', '2018-E-c-En-test.txt'),
        },
        'EI.oc': {
            'train': {
                'anger': os.path.join(base_dir, 'EI-oc-En', 'train', 'EI-oc-En-anger-train.txt'),
                'fear': os.path.join(base_dir, 'EI-oc-En', 'train', 'EI-oc-En-fear-train.txt'),
                'joy': os.path.join(base_dir, 'EI-oc-En', 'train', 'EI-oc-En-joy-train.txt'),
                'sadness': os.path.join(base_dir, 'EI-oc-En', 'train', 'EI-oc-En-sadness-train.txt'),
            },
            'dev': {
                'anger': os.path.join(base_dir, 'EI-oc-En', 'dev', '2018-EI-oc-En-anger-dev.txt'),
                'fear': os.path.join(base_dir, 'EI-oc-En', 'dev', '2018-EI-oc-En-fear-dev.txt'),
                'joy': os.path.join(base_dir, 'EI-oc-En', 'dev', '2018-EI-oc-En-joy-dev.txt'),
                'sadness': os.path.join(base_dir, 'EI-oc-En', 'dev', '2018-EI-oc-En-sadness-dev.txt'),
            },
            'test': {
                'anger': os.path.join(base_dir, 'EI-oc-En', 'test', '2018-EI-oc-En-anger-test.txt'),
                'fear': os.path.join(base_dir, 'EI-oc-En', 'test', '2018-EI-oc-En-fear-test.txt'),
                'joy': os.path.join(base_dir, 'EI-oc-En', 'test', '2018-EI-oc-En-joy-test.txt'),
                'sadness': os.path.join(base_dir, 'EI-oc-En', 'test', '2018-EI-oc-En-sadness-test.txt'),
            },
        },
        'EI.reg': {
            'train': {
                'anger': os.path.join(base_dir, 'EI-reg-En', 'train', 'EI-reg-En-anger-train.txt'),
                'fear': os.path.join(base_dir, 'EI-reg-En', 'train', 'EI-reg-En-fear-train.txt'),
                'joy': os.path.join(base_dir, 'EI-reg-En', 'train', 'EI-reg-En-joy-train.txt'),
                'sadness': os.path.join(base_dir, 'EI-reg-En', 'train', 'EI-reg-En-sadness-train.txt'),
            },
            'dev': {
                'anger': os.path.join(base_dir, 'EI-reg-En', 'dev', '2018-EI-reg-En-anger-dev.txt'),
                'fear': os.path.join(base_dir, 'EI-reg-En', 'dev', '2018-EI-reg-En-fear-dev.txt'),
                'joy': os.path.join(base_dir, 'EI-reg-En', 'dev', '2018-EI-reg-En-joy-dev.txt'),
                'sadness': os.path.join(base_dir, 'EI-reg-En', 'dev', '2018-EI-reg-En-sadness-dev.txt'),
            },
            'test': {
                'anger': os.path.join(base_dir, 'EI-reg-En', 'test', '2018-EI-reg-En-anger-test.txt'),
                'fear': os.path.join(base_dir, 'EI-reg-En', 'test', '2018-EI-reg-En-fear-test.txt'),
                'joy': os.path.join(base_dir, 'EI-reg-En', 'test', '2018-EI-reg-En-joy-test.txt'),
                'sadness': os.path.join(base_dir, 'EI-reg-En', 'test', '2018-EI-reg-En-sadness-test.txt'),
            },
        },
        'V.oc': {
            'train': os.path.join(base_dir, 'V-oc-En', 'train', '2018-Valence-oc-En-train.txt'),
            'dev': os.path.join(base_dir, 'V-oc-En', 'dev', '2018-Valence-oc-En-dev.txt'),
            'test': os.path.join(base_dir, 'V-oc-En', 'test', '2018-Valence-oc-En-test.txt'),
        },
        'V.reg': {
            'train': os.path.join(base_dir, 'V-reg-En', 'train', '2018-Valence-reg-En-train.txt'),
            'dev': os.path.join(base_dir, 'V-reg-En', 'dev', '2018-Valence-reg-En-dev.txt'),
            'test': os.path.join(base_dir, 'V-reg-En', 'test', '2018-Valence-reg-En-test.txt'),
        },
    }[task]


def _load_EI_reg(files):
    data = {'train': dict(), 'dev': dict(), 'test': dict()}
    for dataset_type in ['train', 'dev', 'test']:
        ids = set()
        for file_name in files[dataset_type].keys():
            file = files[dataset_type][file_name]
            f = open(file, "rb")
            lines = f.readlines()
            skip_header = True
            for line in lines:
                if skip_header:
                    skip_header = False
                    continue
                tid, text, emotion, rating = line.decode().strip().split('\t')
                tid = '{}-{}'.format(tid, emotion)
                rating = float(rating) if rating != 'NONE' else None
                data[dataset_type][tid] = {'text': text, 'emotion': emotion, 'rating': rating}
                if tid in ids:
                    msg = 'Records with duplicated ID {0} found. Use records with unique ID.'.format(tid)
                    raise ValueError(msg)
                ids.add(tid)
    train_df = pd.DataFrame.from_dict(data['train'], orient='index')
    dev_df = pd.DataFrame.from_dict(data['dev'], orient='index')
    test_df = pd.DataFrame.from_dict(data['test'], orient='index')
    return train_df, dev_df, test_df


def _load_EL_oc(files):
    data = {'train': dict(), 'dev': dict(), 'test': dict()}
    for dataset_type in ['train', 'dev', 'test']:
        ids = set()
        for file_name in files[dataset_type].keys():
            file = files[dataset_type][file_name]
            f = open(file, "rb")
            lines = f.readlines()
            skip_header = True
            for line in lines:
                if skip_header:
                    skip_header = False
                    continue
                tid, text, emotion, rating = line.decode().strip().split('\t')
                tid = '{}-{}'.format(tid, emotion)
                if tid in ids:
                    msg = 'Records with duplicated ID {0} found. Use records with unique ID.'.format(tid)
                    raise ValueError(msg)
                ids.add(tid)
                oc = int(rating.split(':')[0]) if rating != 'NONE' else None
                data[dataset_type][tid] = {'text': text, 'emotion': emotion, 'label': oc}
    train_df = pd.DataFrame.from_dict(data['train'], orient='index')
    dev_df = pd.DataFrame.from_dict(data['dev'], orient='index')
    test_df = pd.DataFrame.from_dict(data['test'], orient='index')
    return train_df, dev_df, test_df


def _load_E_c(path):
    train, dev, test = pd.read_csv(path['train'], sep='\t'), \
                       pd.read_csv(path['dev'], sep='\t'), \
                       pd.read_csv(path['test'], sep='\t')
    return train, dev, test


def _load_V_oc(path):
    train, dev, test = pd.read_csv(path['train'], sep='\t'), \
                       pd.read_csv(path['dev'], sep='\t'), \
                       pd.read_csv(path['test'], sep='\t')
    train = train.drop(['Affect Dimension'], axis=1)
    dev = dev.drop(['Affect Dimension'], axis=1)
    test = test.drop(['Affect Dimension'], axis=1)
    train.columns = ['id', 'text', 'label']
    dev.columns = ['id', 'text', 'label']
    test.columns = ['id', 'text', 'label']
    train['label'] = train['label'].apply(lambda x: int(x.split(':')[0]) if x != 'NONE' else None)
    dev['label'] = dev['label'].apply(lambda x: int(x.split(':')[0]) if x != 'NONE' else None)
    test['label'] = test['label'].apply(lambda x: int(x.split(':')[0]) if x != 'NONE' else None)
    return train, dev, test


def _load_V_reg(path):
    train, dev, test = pd.read_csv(path['train'], sep='\t'), \
                       pd.read_csv(path['dev'], sep='\t'), \
                       pd.read_csv(path['test'], sep='\t')
    train = train.drop(['Affect Dimension'], axis=1)
    dev = dev.drop(['Affect Dimension'], axis=1)
    test = test.drop(['Affect Dimension'], axis=1)
    train.columns = ['id', 'text', 'rating']
    dev.columns = ['id', 'text', 'rating']
    test.columns = ['id', 'text', 'rating']
    return train, dev, test


def load_ait(path='', task='E.c'):
    path = _file_paths(path, task)
    if task == 'EI.oc':
        return _load_EL_oc(path)
    if task == 'EI.reg':
        return _load_EI_reg(path)
    elif task == 'E.c':
        return _load_E_c(path)
    elif task == 'V.oc':
        return _load_V_oc(path)
    elif task == 'V.reg':
        return _load_V_reg(path)
    else:
        return None, None, None
