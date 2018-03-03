import os
import pandas as pd


def load_emoint(input_path=''):
    ids = set()
    files = {
        'train': {
            'anger_train': os.path.join(input_path, 'train\\anger-ratings-0to1.train.txt'),
            'fear_train': os.path.join(input_path, 'train\\fear-ratings-0to1.train.txt'),
            'joy_train': os.path.join(input_path, 'train\\joy-ratings-0to1.train.txt'),
            'sadness_train': os.path.join(input_path, 'train\\sadness-ratings-0to1.train.txt'),
        },
        'dev': {
            'anger_dev': os.path.join(input_path, 'dev.gold\\anger-ratings-0to1.dev.gold.txt'),
            'fear_dev': os.path.join(input_path, 'dev.gold\\fear-ratings-0to1.dev.gold.txt'),
            'joy_dev': os.path.join(input_path, 'dev.gold\\joy-ratings-0to1.dev.gold.txt'),
            'sadness_dev': os.path.join(input_path, 'dev.gold\\sadness-ratings-0to1.dev.gold.txt'),
        },
        'test': {

        }
    }
    data = {'train': dict(), 'dev': dict(), 'test': dict()}
    for dataset_type in ['train', 'dev', 'test']:
        for file_name in files[dataset_type].keys():
            file = files[dataset_type][file_name]
            f = open(file, "rb")
            lines = f.readlines()
            for line in lines:
                id, text, emotion, rating = line.decode().strip().split('\t')
                data[dataset_type][id] = {'text': text, 'emotion': emotion, 'rating': rating}
                if id in ids:
                    msg = 'Records with duplicated ID {0} found. Use records with unique ID.'.format(id)
                    raise ValueError(msg)
                ids.add(id)
    train_df = pd.DataFrame.from_dict(data['train'], orient='index')
    dev_df = pd.DataFrame.from_dict(data['dev'], orient='index')
    test_df = pd.DataFrame.from_dict(data['test'], orient='index')
    return train_df, dev_df, test_df
