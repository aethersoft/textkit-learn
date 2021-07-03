""" Base IO code for all datasets """

import json
import os
import pickle
import shutil
from os import makedirs
from os.path import exists, expanduser, join

import pandas as pd

from tklearn import utils
from tklearn.config import config
from tklearn.utils import dict_normalize

RESOURCE_PATH = config['DEFAULT']['resource_path']

# noinspection SpellCheckingInspection
__all__ = [
    'get_data_home',
    'clear_data_home',
    'download',
    'load_davidson17',
    'load_founta18',
    'load_fdcl18',
    'load_olid',
    'load_dataset',
]

DATA_FILES = dict_normalize({
    'OffensEval2020': {
        'train': {
            'task_a': 'OffensEval2020/train/task_a_distant.tsv',
            'task_b': 'OffensEval2020/train/task_a_distant.tsv',
            'task_c': 'OffensEval2020/train/task_c_distant_ann.tsv'
        }
    },
    'founta18': {
        'retweets': 'founta18/retweets.csv',
        'hatespeech': 'founta18/hatespeech_text_label_vote_RESTRICTED_100K.csv',
    }
})


# noinspection SpellCheckingInspection
def get_data_home(data_home=None):
    """ Return the path of the tklearn data dir.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data dir is set to a folder named '.olang/data' in the
    user home folder.

    Alternatively, it can be set by the 'OLING_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home
        The path to onling data dir.

    Returns
    -------
    data_path
        The path to onling data dir.
    """
    if data_home is None:
        data_home = join(RESOURCE_PATH, 'data')
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


# noinspection SpellCheckingInspection
def clear_data_home(data_home=None):
    """ Delete all the content of the data home cache.

    Parameters
    ----------
    data_home
        The path to tklearn data dir.

    Returns
    -------
    null
        None
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


# noinspection SpellCheckingInspection
def download():
    """ Download data from server to local.

    Returns
    -------
    null
        None
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
    # Download data for of Hate Speech (OffensEval2020) dataset.
    url = 'https://ysenarath.s3.amazonaws.com/dataset/OffensEval2020.zip'
    file_name = 'OffensEval2020.zip'
    utils.download(url, get_data_home(), file_name, unzip=True)
    # Download data for of Hate Speech (founta18) dataset.
    url = 'https://ysenarath.s3.amazonaws.com/dataset/founta18.zip'
    file_name = 'founta18.zip'
    utils.download(url, get_data_home(), file_name, unzip=True)


# noinspection SpellCheckingInspection
def _load_file(module_path, data_file_name, column_names=None):
    """ Loads data from module_path/data/data_file_name.

    Parameters
    ----------
    module_path
        Path to module.

    data_file_name
        Name of csv file to be loaded from module_path/data/data_file_name. For example 'wine_data.csv'.

    column_names

    Returns
    -------
    data : Pandas.DataFrame
        Loads file depending on the file-type.
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


# noinspection SpellCheckingInspection
def load_davidson17(**kwargs):
    """ Load and return the hate-speech (Davidson17) dataset  (classification).

    Please refer to `Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017, May).
    Automated hate speech detection and the problem of offensive language.
    In Eleventh international aaai conference on web and social media.
    ` for more information on DWMW17 dataset.

    Parameters
    ----------
    kwargs

    Returns
    -------
    data : Pandas.DataFrame
        Loads file depending on the file-type.
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


# noinspection SpellCheckingInspection
def _fdcl18_format_tweet(x):
    """Clean the format of FDCL18 dataset text.

    :param x: Input text.
    :return: Reformatted input text.
    """
    try:
        return json.loads('{}{}{}'.format('{', x, '}'))['text'].encode('utf-8').decode('ascii', errors='ignore')
    except:
        return json.loads('{}{}\"{}'.format('{', x, '}'))['text'].encode('utf-8').decode('ascii', errors='ignore')


# noinspection SpellCheckingInspection
def load_fdcl18(**kwargs):
    """ Load and return the hate-speech (fdcl18) dataset  (classification).

    Please refer to `Founta, A. M., Djouvas, C., Chatzakou, D., Leontiadis, I., Blackburn, J., Stringhini, G.,
     ... & Kourtellis, N. (2018, June). Large scale crowdsourcing and characterization of twitter abusive behavior.
      In Twelfth International AAAI Conference on Web and Social Media.` for more information on DWMW17 dataset.

    Parameters
    ----------
    kwargs
        Configurations for loading dataset.

    Returns
    -------
    data : Pandas.DataFrame
        Loads file depending on the file-type.
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


# noinspection SpellCheckingInspection
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


# noinspection SpellCheckingInspection
def load_founta18(filename=None):
    """ Load and return the hate-speech (fdcl18) dataset  (classification). [Full Dataset]

    Please refer to `Founta, A. M., Djouvas, C., Chatzakou, D., Leontiadis, I., Blackburn, J., Stringhini, G.,
     ... & Kourtellis, N. (2018, June). Large scale crowdsourcing and characterization of twitter abusive behavior.
      In Twelfth International AAAI Conference on Web and Social Media.` for more information on DWMW17 dataset.


    Parameters
    ----------
    filename
        Name of the file for loading dataset. Not required if it is downloaded to default location.

    Returns
    -------
    data : Pandas.DataFrame
        Loads file depending on the file-type.
    """
    if filename is None:
        filename = 'hatespeech'
    path = os.path.join(get_data_home(), DATA_FILES[f'founta18.{filename}'])
    df = pd.read_csv(path, sep='\t', header=None, names=['text', 'label', 'vote'])
    return df


# noinspection SpellCheckingInspection
def load_offenseval20(task='task_a', split='train'):
    """ Loads competition dataset for Offens Eval 2020 dataset.

    Parameters
    ----------
    task
        Name of the task to load. Supported [task_a, task_b, task_c, ]

    split
        Name of the split to load. Supported [train, ].

    Returns
    -------
    data : Pandas.DataFrame
        Loads file depending on the file-type.
    """
    return _load_file(get_data_home(), DATA_FILES[f'OffensEval2020.{split}.{task}'])


# noinspection SpellCheckingInspection
def load_dataset(name, **kwargs):
    """ Loads and returns the dataset with the provided name.

    Parameters
    ----------
    name
        Name of the dataset.

    kwargs
        Configurations for loading dataset.¬

    Returns
    -------
    data : Pandas.DataFrame
        Loads file depending on the file-type.
    """
    if name.lower().startswith('davidson17') or name.lower().startswith('dwmw17'):
        df = load_davidson17(**kwargs)
    elif name.lower().startswith('olid'):
        df = load_olid(**kwargs)
    elif name.lower().startswith('offenseval20'):
        df = load_offenseval20(**kwargs)
    elif name.lower().startswith('founta18') or name.lower().startswith('fdcl18'):
        df = load_founta18(**kwargs)
    elif name.lower().startswith('fdcl18-deprecated'):
        df = load_fdcl18(**kwargs)
    else:
        raise ValueError('Invalid dataset name. Please enter valid name.')
    return df
