import csv
import gzip
import json
import os
import re
from collections import defaultdict, Counter

import nltk
import numpy as np
from sklearn.preprocessing import FunctionTransformer

from tklearn.text.tokens import bigrams
from tklearn.utils.collections import merge_dicts
from tklearn.utils.resource import resource_path

__all__ = ['LexiconFeaturizer']


# ======================================================================================================================
# Embedding Featurizer--------------------------------------------------------------------------------------------------

class ExtractEmbedding:
    def __init__(self, lexicon_path, word_first=False, leave_head=False):
        self.lexicon_path = lexicon_path
        self.word_first = word_first
        self.leave_head = leave_head
        self.embedding_map = self.load_embeddings()

    def __call__(self, tokens):
        """
        Averaging word embeddings
        """
        if len(self.embedding_map.keys()) > 0:
            dim = len(list(self.embedding_map.values())[0])
        else:
            dim = 0
        sum_vec = np.zeros(shape=(dim,))
        for token in tokens:
            if token in self.embedding_map:
                vec = self.embedding_map[token]
                assert len(vec) == dim, 'Invalid length for embedding provided. Observed {} expected {}' \
                    .format(len(vec), dim)
                sum_vec = sum_vec + vec
        denom = len(tokens)
        sum_vec = sum_vec / denom
        return sum_vec

    def load_embeddings(self):
        """
        Creates a map from words to word embeddings
        :return: embedding map
        """
        with gzip.open(self.lexicon_path, 'rb') as f:
            lines = f.read().decode('utf8').splitlines()
        if self.leave_head:
            lines = lines[1:]
        lexicon_map = {}
        for l in lines:
            splits = re.split('[\t ]', l)
            if self.word_first:
                lexicon_map[splits[0]] = np.asarray(splits[1:], dtype='float32')
            else:
                lexicon_map[splits[-1]] = np.asarray(splits[:-1], dtype='float32')
        return lexicon_map


# ======================================================================================================================
# Lexicon Featurizer----------------------------------------------------------------------------------------------------

class SentiWordnetScorer:
    def __init__(self, lexicon_path):
        self.lexicon_path = lexicon_path
        self.lexicon_map_ = self.load_lexicon()

    def __call__(self, tokens):
        """
        This function returns sum of intensities of positive and negative tokens
        :param text: text to featurize
        :param tokenizer: tokenizer to tokenize text
        """
        positive_score, negative_score = 0.0, 0.0
        for token in tokens:
            token = token.lower()
            if token in self.lexicon_map_:
                if self.lexicon_map_[token] >= 0:
                    positive_score += self.lexicon_map_[token]
                else:
                    negative_score += self.lexicon_map_[token]
        return [positive_score, negative_score]

    def load_lexicon(self):
        """Creates a map from lexicons to either positive or negative
         :param lexicon_path path of lexicon file (in gzip format)
         """
        with gzip.open(self.lexicon_path, 'rb') as f:
            lines = f.read().decode('utf8').splitlines()
            lexicon_map = defaultdict(float)
            for l in lines:
                if l.strip().startswith('#'):
                    continue
                splits = l.split('\t')
                # positive score - negative score
                score = float(splits[2]) - float(splits[3])
                words = splits[4].split(" ")
                # iterate through all words
                for word in words:
                    word, rank = word.split('#')
                    # scale scores according to rank
                    # more popular => less rank => high weight
                    lexicon_map[word] += (score / float(rank))
            return lexicon_map


class PolarityCounter:
    def __init__(self, *lexicon_paths):
        self.lexicon_paths = list(lexicon_paths)
        self.lexicon_map = self.load_lexicons()

    def __call__(self, tokens):
        """
        This function returns count of positive and negative tokens
        :param tokens: tokens to featurize
        :param tokenizer: tokenizer to tokenize text
        """
        positive_count, negative_count = 0.0, 0.0
        for token in tokens:
            if token in self.lexicon_map:
                if self.lexicon_map[token] == 'positive':
                    positive_count += 1
                else:
                    negative_count += 1
        return [positive_count, negative_count]

    def load_lexicons(self):
        """
        Creates a map from lexicons to either positive or negative
        :param lexicon_path path of lexicon file (in gzip format)
        """
        temp = []
        for lexicon_path in self.lexicon_paths:
            with gzip.open(lexicon_path, 'rb') as f:
                lines = f.read().decode('utf8').splitlines()
                lm = {}
                for l in lines:
                    splits = l.split('\t')
                    lm[splits[0]] = splits[1]
                temp.append(lm)
        return merge_dicts(*temp)


class PolarityScorer:
    """
    Gives the input tokens a polarity score based on lexicons
    """

    def __init__(self, *lexicon_paths):
        lexicon_maps = []
        for lexicon_path in lexicon_paths:
            with gzip.open(lexicon_path, 'rb') as f:
                lines = f.read().decode('utf8').splitlines()
                lexicon_map = {}
                for l in lines:
                    splits = l.split('\t')
                    lexicon_map[splits[0]] = float(splits[1])
                lexicon_maps.append(lexicon_map)
        self.lexicon_map_ = merge_dicts(*lexicon_maps)

    def __call__(self, tokens, bigram=True):
        unigrams = tokens
        if bigram:
            bt = bigrams(tokens)
            scores = [x + y for x, y in zip(self.extract_scores(unigrams), self.extract_scores(bt))]
        else:
            scores = self.extract_scores(unigrams)
        return scores

    def extract_scores(self, tokens):
        """This function returns sum of intensities of positive and negative tokens
        :param tokens tokens to featurize
        """
        positive_score, negative_score = 0.0, 0.0
        for token in tokens:
            token = token.lower()
            if token in self.lexicon_map_:
                if self.lexicon_map_[token] >= 0:
                    positive_score += self.lexicon_map_[token]
                else:
                    negative_score += self.lexicon_map_[token]
        return [positive_score, negative_score]


class SentimentRanking:
    """
    Ranks the sentiment based on three keys [Negative, Positive, Neutral]
    """

    def __init__(self, fid, lexicons_path):
        rows = csv.DictReader(open(lexicons_path, 'r', encoding="utf8"), )
        emoji_map = {}
        for row in rows:
            emoji_map[row['Emoji']] = [
                float(row['Negative']) / float(row['Occurrences']),
                float(row['Positive']) / float(row['Occurrences']),
                float(row['Neutral']) / float(row['Occurrences'])
            ]
        self.emoji_map = emoji_map
        self.fid = fid
        self.features = [self.fid + '_' + x for x in ['Negative', 'Neutral', 'Positive']]

    def __call__(self, tokens):
        sum_vec = [0.0] * len(self.features)
        for token in tokens:
            if token in self.emoji_map:
                sum_vec = [a + b for a, b in zip(sum_vec, self.emoji_map[token])]
        return sum_vec


class NegationCounter:
    """
    Counts negations and return the count
    """

    def __init__(self, lexicon_path):
        self.lexicon_path = lexicon_path
        self.lexicon_map = self.load_lexicon()

    def __call__(self, tokens):
        count = 0
        for token in tokens:
            if token in self.lexicon_map:
                count += 1
        return [count]

    def load_lexicon(self):
        """
        Creates a map from lexicons to either positive or negative
        :param lexicon_path path of lexicon file (in gzip format)
        """
        with gzip.open(self.lexicon_path, 'rb') as f:
            lines = f.read().splitlines()
            lexicon_map = defaultdict(int)
            for l in lines:
                splits = l.decode('utf-8').split('\t')
                lexicon_map[splits[0]] += 1
        return lexicon_map


class EmotionLexiconScorer:

    def __init__(self, lexicon_path):
        self.lexicon_path = lexicon_path
        self.emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
                         'negative', 'positive', 'sadness', 'surprise', 'trust']
        self.lexicon_map = self.load_lexicon()

    def __call__(self, tokens):
        """This function returns score of tokens belonging to different emotions
        :param text: text to featurize
        :param tokenizer: tokenizer to tokenize text
        """
        sum_vec = [0.0] * len(self.emotions)
        for token in tokens:
            if token in self.lexicon_map:
                sum_vec = [a + b for a, b in zip(sum_vec, self.lexicon_map[token])]
        return sum_vec

    def get_missing(self, text, tokenizer):
        tokens = tokenizer.tokenize(text)
        tc, mc = 0.0, 0.0
        for token in tokens:
            tc += 1
            if not token in self.lexicon_map:
                mc += 1
        if tc == 0:
            return 1.0
        else:
            # print("Total: {}, Missing: {}".format(tc, mc * 1.0 / tc * 1.0))
            return mc * 1.0 / tc * 1.0

    def load_lexicon(self):
        """
        Creates a map from lexicons to either positive or negative
        :param lexicon_path path of lexicon file (in gzip format)
        """
        with gzip.open(self.lexicon_path, 'rb') as f:
            lines = f.read().splitlines()
            lexicon_map = defaultdict(list)

            for l in lines[1:]:
                splits = l.decode('utf-8').split('\t')
                lexicon_map[splits[0]] = [float(num) for num in splits[1:]]

        return lexicon_map


class SentiStrengthScorer:
    """SentiStrength Featurizer"""

    def __init__(self, jar_path, dir_path):
        """
        Initialize SentiStrength featurizer
        :param jar_path:
        :param dir_path:
        """
        self.jar_path = jar_path
        self.dir_path = dir_path

        if 'CLASSPATH' in os.environ:
            os.environ['CLASSPATH'] += ":" + jar_path
        else:
            os.environ['CLASSPATH'] = jar_path

        # Add jar to class path
        # Create and initialize the SentiStrength class
        self.load_success = False
        try:
            from jnius import autoclass

            self.senti_obj = autoclass('uk.ac.wlv.sentistrength.SentiStrength')()
            self.senti_obj.initialise(["sentidata", dir_path, "trinary"])
            self.load_success = True
        except ImportError:
            pass

    def __call__(self, tokens):
        """This function returns sum of intensities of positive and negative tokens
        :param text text to featurize
        :param tokenizer tokenizer to tokenize text
        """
        if not self.load_success:
            return [0.0, 0.0]
        data = '+'.join(tokens).encode('utf-8').decode("utf-8", "ignore")
        score = self.senti_obj.computeSentimentScores(data)
        splits = score.rstrip().split(' ')
        return [float(splits[0]), float(splits[1])]


# ======================================================================================================================
# Lingusistic Featurizer------------------------------------------------------------------------------------------------

class LIWCTrie:
    def __init__(self):
        self.root = dict()
        self.end = '$'
        self.cont = '*'

    def insert(self, word, categories):
        """
        Insert liwc word and categories in trie
        :param word: word to insert
        :param categories: liwc categories
        :return: None
        """
        if len(word) == 0:
            return
        curr_dict = self.root
        for letter in word[:-1]:
            curr_dict = curr_dict.setdefault(letter, {})
        if word[-1] != self.cont:
            curr_dict = curr_dict.setdefault(word[-1], {})
            curr_dict[self.end] = categories
        else:
            curr_dict[self.cont] = categories

    def in_trie(self, word):
        """
        Query if a word is in trie or not
        :param word: query word
        :return: True if word is present otherwise False
        """
        curr_dict = self.root
        for letter in word:
            if letter in curr_dict:
                curr_dict = curr_dict[letter]
            elif self.cont in curr_dict:
                return True
            else:
                return False
        if self.cont in curr_dict or self.end in curr_dict:
            return True
        else:
            return False

    def get(self, word):
        """
        get value stored against word
        :param word: word to search
        :return: list of categories if word is in query otherwise None
        """
        curr_dict = self.root
        for letter in word:
            if letter in curr_dict:
                curr_dict = curr_dict[letter]
            elif self.cont in curr_dict:
                return curr_dict[self.cont]
            else:
                return None
        if self.cont in curr_dict or self.end in curr_dict:
            if self.cont in curr_dict:
                return curr_dict[self.cont]
            else:
                return curr_dict[self.end]
        else:
            return None

    def __getitem__(self, item):
        return self.get(item)

    def __contains__(self, item):
        return self.in_trie(item)


class LIWCExtractor:
    def __init__(self, lexicons_path):
        """
        Creates a LIWC Lexicon Featurizer

        :param caching: whether to use in memory cache. useful in cross-validation tasks
        """
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        self.categories, self.liwc_trie = self.create_lexicon_mapping(lexicons_path)
        self.features = ['total_word_count', 'avg_sentence_length', 'dictionary_words',
                         '>six_letter_words', 'numerals'] + list(self.categories) + [x[0] for x in self.punctuations()]
        self.features.sort()

    def __call__(self, tokens):
        """
        This function returns count of positive and negative tokens
        :param tokens: tokens to featurize
        """
        liwc = {}

        text = ' '.join(tokens)
        num_capital_words = len(re.findall(r"[A-Z]['A-Z]*", text))
        words = re.findall(r"[a-z]['a-z]*", text.lower())

        text = text.lower()
        num_words = len(words)

        # text level features
        liwc['total_word_count'] = num_words
        liwc['num_capital_words'] = self.percentage(num_capital_words, num_words)
        if len(nltk.sent_tokenize(text)) > 0:
            liwc['avg_sentence_length'] = np.mean([self.number_of_words(sent)
                                                   for sent in nltk.sent_tokenize(text)])
        else:
            liwc['avg_sentence_length'] = 1.0
        liwc['>six_letter_words'] = self.percentage(sum([1 for x in words if len(x) >= 6]), num_words)
        liwc['dictionary_words'] = self.percentage(sum([1 for x in words if x in self.liwc_trie]), num_words)
        liwc['numerals'] = self.percentage(sum([1 for x in words if x.isdigit()]), num_words)

        for cat in self.categories:
            liwc[cat] = 0.0

        # categorical features
        for token in tokens:
            if token in self.liwc_trie:
                for cat in self.liwc_trie[token]:
                    liwc[cat] += 1.0

        for cat in self.categories:
            liwc[cat] = self.percentage(liwc[cat], num_words)

        self.set_punctuation_counts(text, liwc)

        return [liwc[x] for x in self.features]

    def set_punctuation_counts(self, text, liwc):
        character_counts, counts = Counter(text), {}
        for name, chars in self.punctuations():
            counts[name] = sum(character_counts[char] for char in chars)
        counts['Parenth'] /= 2.0
        counts['AllPct'] = sum(counts[name] for name, _ in self.punctuations())
        for x, y in counts.items():
            liwc[x] = self.percentage(y, liwc['total_word_count'])

    @staticmethod
    def number_of_words(text):
        return len(re.findall(r"[a-z]['a-z]*", text.lower()))

    @staticmethod
    def percentage(a, b):
        # return a
        return (a * 100.0) / (b * 1.0 + 1.0)

    @staticmethod
    def punctuations():
        return [('Period', '.'), ('Comma', ','), ('Colon', ':'), ('SemiC', ';'), ('QMark', '?'), ('Exclam', '!'),
                ('Dash', '-'), ('Quote', '"'), ('Apostro', "'"), ('Parenth', '()[]{}'),
                ('OtherP', '#$%&*+-/<=>@\\^_`|~')]

    @staticmethod
    def create_lexicon_mapping(lexicon_path):
        liwc_trie = LIWCTrie()
        data = open(lexicon_path).read()
        splits = data.split('%')
        categories = dict([(x.split('\t')[0], x.split('\t')[1]) for x in splits[1].strip().splitlines()])
        for x in splits[2].strip().splitlines():
            try:
                pair = (x.split('\t')[0], [categories[y] for y in x.strip().split('\t')[1:]])
                liwc_trie.insert(pair[0], pair[1])
            except Exception as ex:
                pass
        return categories.values(), liwc_trie


# ======================================================================================================================
# Featurizer class -----------------------------------------------------------------------------------------------------


class LexiconFeaturizer(FunctionTransformer):
    """
    The lexicon featurizer converts a list of tokenized documents into a vector,
     in the format used for classification.
     The actual feature to be used is provided as parameter
    """

    __mem_cache = defaultdict(dict)

    def __init__(self, lexicons=None, caching=None):
        """
        Creates a LexiconFeaturizer

        :param caching: whether to use in memory cache. useful in cross-validation tasks
        """
        super(LexiconFeaturizer, self).__init__(self._get_lexicon_features, validate=False, )
        self.caching = caching
        self.lexicons = lexicons

    def _get_lexicon_features(self, seq):
        """
        Transforms a sequence of token input to a vector based on lexicons.

        :param seq: Sequence of token inputs
        :return: a list of feature vectors extracted from the sequence of texts in the same order
        """
        if isinstance(self.lexicons, str):
            return self._extract_features(seq, self.lexicons)
        else:
            outs = [self._extract_features(seq, f) for f in self.lexicons]
            return np.concatenate(outs, axis=1)

    def _extract_features(self, seq, feature):
        outs = []
        extract_features = _get_featurizer(feature)
        for tokens in seq:
            text = ' '.join(tokens)
            if self.caching and text in LexiconFeaturizer.__mem_cache[feature]:
                temp = LexiconFeaturizer.__mem_cache[feature][text]
            else:
                temp = extract_features(tokens)
            outs.append(np.array(temp))
            if self.caching and text not in LexiconFeaturizer.__mem_cache[feature]:
                LexiconFeaturizer.__mem_cache[feature][text] = temp
        outs = np.array(outs)
        if len(outs.shape) == 1:
            outs = np.reshape(outs, (-1, 1))
        return outs


def _get_lexicon(name):
    resources = json.load(open(resource_path('resources.json')))
    lexicons = ['lexicons'] + resources['lexicons'][name]
    path = resource_path(*lexicons)
    return path


def _get_featurizer(name):
    """
    Gets resources from resource path.
     Resource path should contain json file indicating the resources and how to access them.
    :param name: name of the lexicon resource
    :return: path to lexicon
    """
    resources = json.load(open(resource_path('resources.json')))
    featurizer = resources['featurizers'][name]['class']
    lexicons = resources['featurizers'][name]['lexicons']
    lexicons = [_get_lexicon(l) for l in lexicons]
    if featurizer == 'PolarityCounter':
        return PolarityCounter(*lexicons)
    elif featurizer == 'SentiWordnetScorer':
        return SentiWordnetScorer(*lexicons)
    elif featurizer == 'PolarityScorer':
        return PolarityScorer(*lexicons)
    elif featurizer == 'SentimentRanking':
        fid = resources['featurizers'][name]['id']
        return SentimentRanking(fid, *lexicons)
    elif featurizer == 'LIWCExtractor':
        return LIWCExtractor(*lexicons)
    elif featurizer == 'ExtractEmbedding':
        return ExtractEmbedding(*lexicons)
    elif featurizer == 'NegationCounter':
        return NegationCounter(*lexicons)
    elif featurizer == 'EmotionLexiconScorer':
        return EmotionLexiconScorer(*lexicons)
    elif featurizer == 'SentiStrengthScorer':
        return SentiStrengthScorer(*lexicons)
    else:
        raise ModuleNotFoundError('No module named {}'.format(featurizer))