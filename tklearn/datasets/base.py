"""
Base IO code for all datasets
"""
import json
import pickle
import shutil
from os import makedirs
from os.path import exists, expanduser, join

import pandas as pd
from tklearn.configs import configs

from tklearn import utils

__all__ = [
    'get_data_home',
    'clear_data_home',
    'download_data',
    'load_dwmw17',
    'load_fdcl18',
    'load_olid',
    'load_dataset',
]


def get_data_home(data_home=None):
    """Return the path of the olang data dir.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data dir is set to a folder named '.olang/data' in the
    user home folder.

    Alternatively, it can be set by the 'OLING_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    :param data_home: The path to onling data dir.
    :return: The path to onling data dir.
    """
    if data_home is None:
        data_home = join(configs['RESOURCE_PATH'], 'data')
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.

    :param data_home: The path to onling data dir.
    :return: None
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def download_data():
    """Download data from server to local.

    :return: None
    """
    # Download data for of Hate Speech (dwmw17) dataset.
    data_home = join(get_data_home(), 'dwmw17')
    url = 'https://codeload.github.com/t-davidson/hate-speech-and-offensive-language/zip/master'
    file_name = 'hate-speech-and-offensive-language-master.zip'
    utils.download(url, data_home, file_name, unzip=True)
    # Download Frame Features of Hate Speech (dwmw17) dataset.
    url = 'https://www.dropbox.com/s/4jlsdn7gyh8ra63/all_frames_hatespeechtwitter_davidson.pickle?dl=1'
    file_name = 'davidson_frame_data.pickle'
    utils.download(url, data_home, file_name)
    # Download data for of Hate Speech (fdcl18) dataset.
    data_home = join(get_data_home(), 'fdcl18')
    url = 'https://www.dropbox.com/s/5hvlefwg7m8v4t9/hatespeechtwitter.xlsx?dl=1'
    file_name = 'hatespeechtwitter.xlsx'
    utils.download(url, data_home, file_name)
    # Download Frame Features of Hate Speech (fdcl18) dataset.
    url = 'https://www.dropbox.com/s/l20ydu0mvf38hlv/all_frames_hatespeechtwitter.pickle?dl=1'
    file_name = 'founta_frame_data.pickle'
    utils.download(url, data_home, file_name)
    # ownload data for of Hate Speech (OLIDv1.0-dataset) dataset.
    data_home = join(get_data_home(), 'olid_v1.0')
    url = 'https://sites.google.com/site/offensevalsharedtask/olid/OLIDv1.0.zip?attredirects=0&d=1'
    file_name = 'OLIDv1.0.zip'
    utils.download(url, data_home, file_name, unzip=True)


def _load_file(module_path, data_file_name, column_names=None):
    """Loads data from module_path/data/data_file_name.

    :param module_path: The module path.
    :param data_file_name: Name of csv file to be loaded from
        module_path/data/data_file_name. For example 'wine_data.csv'.
    :return: Pandas.DataFrame containing data.
    """
    if data_file_name.endswith('.csv'):
        return pd.read_csv(join(module_path, data_file_name), names=column_names)
    if data_file_name.endswith('.tsv'):
        return pd.read_csv(join(module_path, data_file_name), sep='\t', names=column_names)
    if data_file_name.endswith('.xlsx'):
        return pd.read_excel(join(module_path, data_file_name))
    if data_file_name.endswith('.json'):
        return pd.read_json(join(module_path, data_file_name))
    elif data_file_name.endswith('.pickle'):
        with open(join(module_path, data_file_name), 'rb') as f:
            return pickle.load(f)
    else:
        raise TypeError('invalid file: %s' % join(module_path, data_file_name))


def load_dwmw17(**kwargs):
    """Load and return the hate-speech (dwmw17) dataset  (classification).

    Please refer to `Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017, May).
    Automated hate speech detection and the problem of offensive language.
    In Eleventh international aaai conference on web and social media.
    ` for more information on DWMW17 dataset.

    :param kwargs: Configurations for loading dataset.
    :return: Pandas.DataFrame containing features.
    """
    data_home = join(get_data_home(), 'dwmw17')
    df = _load_file(join(data_home, 'hate-speech-and-offensive-language-master', 'data'), 'labeled_data.csv')
    jsf = _load_file(data_home, 'davidson_frame_data.pickle')
    df['framenet'] = list(map(lambda x: ' '.join([' '.join(f['framenet']) for f in x if 'framenet' in f]), jsf))
    df['propbank'] = list(map(lambda x: ' '.join([f['propbank'] for f in x if 'propbank' in f]), jsf))
    df.rename(columns={'Unnamed: 0': 'id', 'class': 'label'}, inplace=True)
    if 'remove_null' not in kwargs or kwargs['remove_null'] is not False:
        df = df[df.label.notnull()]
        df = df.reset_index()
    if 'num_classes' in kwargs and kwargs['num_classes'] == 2:
        df.label = df.label != 2
    return df


def _fdcl18_format_tweet(x):
    """Clean the format of FDCL18 dataset text.

    :param x: Input text.
    :return: Reformatted input text.
    """
    try:
        return json.loads('{}{}{}'.format('{', x, '}'))['text'].encode('utf-8').decode('ascii', errors='ignore')
    except:
        return json.loads('{}{}\"{}'.format('{', x, '}'))['text'].encode('utf-8').decode('ascii', errors='ignore')


def load_fdcl18(**kwargs):
    """Load and return the hate-speech (fdcl18) dataset  (classification).

    Please refer to `Founta, A. M., Djouvas, C., Chatzakou, D., Leontiadis, I., Blackburn, J., Stringhini, G.,
     ... & Kourtellis, N. (2018, June). Large scale crowdsourcing and characterization of twitter abusive behavior.
      In Twelfth International AAAI Conference on Web and Social Media.` for more information on DWMW17 dataset.

    :param kwargs: Configurations for loading dataset.
    :return: Pandas.DataFrame containing features.
    """
    data_home = join(get_data_home(), 'fdcl18')
    df = _load_file(data_home, 'hatespeechtwitter.xlsx')
    jsf = _load_file(data_home, 'founta_frame_data.pickle')
    df['framenet'] = list(map(lambda x: ' '.join([' '.join(f['framenet']) for f in x if 'framenet' in f]), jsf))
    df['propbank'] = list(map(lambda x: ' '.join([f['propbank'] for f in x if 'propbank' in f]), jsf))
    df = df.drop('Unnamed: 1', axis=1)  # Remove UNK
    df.rename(columns={'ID': 'id', 'CLASS': 'label', 'TWEET': 'tweet'}, inplace=True)
    df['tweet'] = df.tweet.apply(_fdcl18_format_tweet)
    if 'remove_null' not in kwargs or kwargs['remove_null'] is not False:
        df = df[df.label.notnull()]
        df = df.reset_index()
    if 'num_classes' in kwargs and kwargs['num_classes'] == 2:
        df['label'] = df.label.isin(['abusive', 'hateful'])
    return df


def load_olid(version=1.0, task='subtask_a', split='train'):
    data_home = join(get_data_home(), 'olid_v%1.1f' % version)
    if split == 'train':
        ds = _load_file(data_home, 'olid-training-v1.0.tsv')
        return ds.loc[:, ['id', 'tweet', task]]
    elif split == 'test':
        if task == 'subtask_a':
            tweets = _load_file(data_home, 'testset-levela.tsv')
            labels = _load_file(data_home, 'labels-levela.csv', column_names=['id', 'subtask_a'])
            df = tweets.merge(labels, on='id')
            return df
        elif task == 'subtask_b':
            tweets = _load_file(data_home, 'testset-levelb.tsv')
            labels = _load_file(data_home, 'labels-levelb.csv', column_names=['id', 'subtask_b'])
            df = tweets.merge(labels, on='id')
            return df
        elif task == 'subtask_c':
            tweets = _load_file(data_home, 'testset-levelc.tsv')
            labels = _load_file(data_home, 'labels-levelc.csv', column_names=['id', 'subtask_c'])
            df = tweets.merge(labels, on='id')
            return df
    raise ValueError('Invalid parameters for OLIDv1.0 dataset loader. Please recheck the used parameters.')


def load_dataset(name, **kwargs):
    """Loads and returns the dataset with the provided name.

    :param name: Name of the dataset.
    :param kwargs: Configurations for loading dataset.
    :return: Pandas.DataFrame containing features.
    """
    if name.lower().startswith('fdcl18'):
        df = load_fdcl18(**kwargs)
    elif name.lower().startswith('dwmw17'):
        df = load_dwmw17(**kwargs)
    elif name.lower().startswith('olid'):
        df = load_olid(**kwargs)
    else:
        raise ValueError('Invalid dataset name. Please enter valid name.')
    return df
