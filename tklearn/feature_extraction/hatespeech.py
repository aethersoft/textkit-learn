import json
from builtins import classmethod
from importlib import import_module
from os.path import join
from typing import Text, Iterable

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer

from tklearn.configs import configs

__all__ = [
    'download_hatebase',
    'load_hatebase',
    'HatebaseVectorizer'
]


def download_hatebase(resource_home=None):
    """Downloads hatebase to the resource folder. Perquisite required: `hatebase`.

    :return: Nothing
    """
    HatebaseAPI = import_module('HatebaseAPI', 'hatebase')
    key = input('Please enter your api key for https://hatebase.org/: ')
    hatebase = HatebaseAPI({"key": key})
    filters = {"language": "eng"}
    format = "json"
    # initialize list for all vocabulary entry dictionaries
    en_vocab = {}
    response = hatebase.getVocabulary(filters=filters, format=format)
    pages = response["number_of_pages"]
    # fill the vocabulary list with all entries of all pages
    # this might take some time...
    for page in range(1, pages + 1):
        filters["page"] = str(page)
        response = hatebase.getVocabulary(filters=filters, format=format)
        result = response["result"]
        en_vocab[result['term']] = result
    # Save file in the path
    if resource_home:
        resource_path = join(resource_home, 'hatebase_vocab_en.json')
    else:
        resource_path = join(configs['OLANG_PATH'], 'resources', 'hatebase_vocab_en.json')
    with open(resource_path, 'w', encoding='utf-8') as json_file:
        json.dump(en_vocab, json_file)


def load_hatebase(resource_home=None):
    if resource_home:
        resource_path = join(resource_home, 'hatebase_vocab_en.json')
    else:
        resource_path = join(configs['OLANG_PATH'], 'resources', 'hatebase_vocab_en.json')
    with open(resource_path, 'r', encoding='utf-8') as json_file:
        en_vocab = json.load(json_file)
    return en_vocab


class HatebaseVectorizer(TransformerMixin, BaseEstimator):
    def __init__(self, features=None, tokenizer=None, resource_home=None):
        self.features = features
        self.tokenizer = tokenizer
        self.resource_home = resource_home
        # Initialize
        if self.features is None:
            self.features = ['term']
        self.feature_vectors, self.index = self._prepare_features(load_hatebase(self.resource_home), self.features)
        self.dims = self.feature_vectors.shape[1]
        self.tokenize = self.tokenizer if self.tokenizer else self._whitespace_tokenize

    def fit(self, X, y=None, **kwargs):
        """Included for compatibility with the interface of `TransformerMixin`.

        :param X: Input features.
        :param y: Input labels.
        :return: `self`.
        """
        return self

    def transform(self, X):
        """Extract features from the input array-like.

        :param X: An array-like of sentences to extract Hatebase features.
        :return: Hatebase features.
        """
        features = [self._extract_features(x) for x in X]
        return np.array(features)

    def _preprocess(self, text: Text) -> pd.Series:
        """Preprocess and tokenize input text.

        :param text: An input text.
        :return: Preprocessed and tokenized sentences.
        """
        for v in self.index:
            if '_' in v:
                text = text.replace(v.replace('_', ' '), v)
        return pd.Series(self.tokenize(text))

    def _extract_features(self, text: Text) -> np.ndarray:
        """Extracts features from Text input.

        :param text: Input text.
        :return: `pandas.DataFrame` of Features.
        """
        tokens = self._preprocess(text)
        feature_mtx = [np.zeros(self.dims)]
        for v in tokens:
            if v in self.index:
                feature_mtx.append(self.feature_vectors.loc[self.index[v]].tolist())
        return np.mean(feature_mtx, axis=0)

    @classmethod
    def _whitespace_tokenize(cls, text):
        """Default Tokenizer"""
        return text.split(' ')

    @classmethod
    def _prepare_features(cls, dataset, features):
        """Prepares features for each term

        :param dataset: Hatebase dataset.
        :return: Feature-map, and word index mapping.
        """
        index = {r['term'].replace(' ', '_'): r['vocabulary_id'] for _, r in dataset.items()}
        dataset = pd.DataFrame([v for (k, v) in dataset.items()], index=index)
        dfs = []
        if 'term' in features:
            dfs.append(cls._count_vectorize(dataset.term, 'term_', dataset.vocabulary_id))
        if 'hateful_meaning' in features:
            dfs.append(cls._count_vectorize(dataset.hateful_meaning, 'hmeam_', dataset.vocabulary_id))
        if 'nonhateful_meaning' in features:
            dfs.append(cls._count_vectorize(dataset.nonhateful_meaning, 'nmeam_', dataset.vocabulary_id))
        if 'is_unambiguous' in features:
            dfs.append(cls._label_binarizer(dataset.is_unambiguous, 'iunmb_', dataset.vocabulary_id))
        if 'is_unambiguous_in' in features:
            dfs.append(cls._count_vectorize(dataset.is_unambiguous_in, 'unmbn_', dataset.vocabulary_id))
        if 'average_offensiveness' in features:
            dfs.append(cls._discretize(dataset.average_offensiveness, 'avgeff_', dataset.vocabulary_id))
        if 'plural_of' in features:
            dfs.append(cls._label_binarizer(dataset.plural_of, 'pluof_', dataset.vocabulary_id))
        if 'variant_of' in features:
            dfs.append(cls._label_binarizer(dataset.variant_of, 'varof_', dataset.vocabulary_id))
        if 'transliteration_of' in features:
            dfs.append(cls._label_binarizer(dataset.transliteration_of, 'traof_', dataset.vocabulary_id))
        if 'is_about_nationality' in features:
            dfs.append(cls._label_binarizer(dataset.is_about_nationality, 'abtnat_', dataset.vocabulary_id))
        if 'is_about_ethnicity' in features:
            dfs.append(cls._label_binarizer(dataset.is_about_ethnicity, 'abteth_', dataset.vocabulary_id))
        if 'is_about_ethnicity' in features:
            dfs.append(cls._label_binarizer(dataset.is_about_religion, 'abtrel_', dataset.vocabulary_id))
        if 'is_about_ethnicity' in features:
            dfs.append(cls._label_binarizer(dataset.is_about_religion, 'abtgen_', dataset.vocabulary_id))
        if 'is_about_sexual_orientation' in features:
            dfs.append(cls._label_binarizer(dataset.is_about_sexual_orientation, 'abtsex_', dataset.vocabulary_id))
        if 'is_about_disability' in features:
            dfs.append(cls._label_binarizer(dataset.is_about_sexual_orientation, 'abtdis_', dataset.vocabulary_id))
        if 'is_about_class' in features:
            dfs.append(cls._label_binarizer(dataset.is_about_sexual_orientation, 'abtcls_', dataset.vocabulary_id))
        return pd.concat(dfs, axis=1), index

    @classmethod
    def _discretize(cls, a: pd.Series, prefix: Text = '', index: Iterable = None) -> pd.DataFrame:
        if index is None:
            index = list(range(len(a)))
        a = a.fillna(a.mean())
        _, bin_edges = np.histogram(a, bins='fd')
        data = np.digitize(a, bin_edges)
        return cls._label_binarizer(pd.Series(data), prefix=prefix, index=index)

    @classmethod
    def _label_binarizer(cls, a: pd.Series, prefix: Text = '', index: Iterable = None) -> pd.DataFrame:
        if index is None:
            index = list(range(len(a)))
        a = a.fillna('_NULL_')
        ohe = LabelBinarizer()
        data = ohe.fit_transform(a)
        columns = [prefix + str(s).lower() for s in ohe.classes_[:data.shape[1]]]
        return pd.DataFrame(data, index=index, columns=columns)

    @classmethod
    def _count_vectorize(cls, a: pd.Series, prefix: Text = '', index: Iterable = None) -> pd.DataFrame:
        if index is None:
            index = list(range(len(a)))
        a = a.fillna('')
        cv = CountVectorizer(binary=True)
        data = cv.fit_transform(a)
        columns = [prefix + s for s in cv.get_feature_names()]
        return pd.DataFrame(data.todense(), index=index, columns=columns)