import os

import pandas as pd


def _file_paths(base_dir, task):
    return {
        'E.c': {
            'train': os.path.join(base_dir, 'E-c-En', 'train', '2018-E-c-En-train.txt'),
            'dev': os.path.join(base_dir, 'E-c-En', 'dev', '2018-E-c-En-dev.txt'),
            'test': None,
        },
        'EI.oc': {
            'train': {
                'anger': os.path.join(base_dir, 'EI-oc-En', 'train', 'EI-oc-En-anger-train.txt'),
                'fear': os.path.join(base_dir, 'EI-oc-En', 'train', 'EI-oc-En-fear-train.txt'),
                'joy': os.path.join(base_dir, 'EI-oc-En', 'train', 'EI-oc-En-joy-train.txt'),
                'sadness': os.path.join(base_dir, 'EI-oc-En', 'train', 'EI-oc-En-sadness-train.txt'),
            },
            'dev': {
                'anger': os.path.join(base_dir, 'EI-oc-En''dev', '2018-EI-oc-En-anger-dev.txt'),
                'fear': os.path.join(base_dir, 'EI-oc-En''dev', '2018-EI-oc-En-fear-dev.txt'),
                'joy': os.path.join(base_dir, 'EI-oc-En', 'dev', '2018-EI-oc-En-joy-dev.txt'),
                'sadness': os.path.join(base_dir, 'EI-oc-En', 'dev', '2018-EI-oc-En-sadness-dev.txt'),
            },
            'test': {},
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
            'test': {},
        },
        'V.oc': {
            'train': os.path.join(base_dir, 'V-oc-En', 'train', '2018-Valence-oc-En-train.txt'),
            'dev': os.path.join(base_dir, 'V-oc-En', 'dev', '2018-Valence-oc-En-dev.txt'),
            'test': None,
        },
        'V.reg': {
            'train': os.path.join(base_dir, 'V-reg-E', 'train', '2018-Valence-reg-En-train.txt'),
            'dev': os.path.join(base_dir, 'V-reg-En', 'dev', '2018-Valence-reg-En-dev.txt'),
            'test': None,
        },
    }[task]


def _load_EI_reg(files):
    ids = set()
    data = {'train': dict(), 'dev': dict(), 'test': dict()}
    for dataset_type in ['train', 'dev', 'test']:
        for file_name in files[dataset_type].keys():
            file = files[dataset_type][file_name]
            f = open(file, "rb")
            lines = f.readlines()
            for line in lines:
                id, text, emotion, rating = line.decode().strip().split('\t')
                data[dataset_type][id] = {'text': text, 'emotion': emotion, 'rating': float(rating)}
                if id in ids:
                    msg = 'Records with duplicated ID {0} found. Use records with unique ID.'.format(id)
                    raise ValueError(msg)
                ids.add(id)
    train_df = pd.DataFrame.from_dict(data['train'], orient='index')
    dev_df = pd.DataFrame.from_dict(data['dev'], orient='index')
    test_df = pd.DataFrame.from_dict(data['test'], orient='index')
    return train_df, dev_df, test_df


def load_ait(path='', task='E.c'):
    assert task == 'E.c', 'Task not supported yet.'
    path = _file_paths(path, task)
    train = pd.read_csv(path['train'], sep='\t')
    dev = pd.read_csv(path['dev'], sep='\t')
    test = None
    return train, dev, test
