import csv
import gzip
import os
import re
from collections import defaultdict, Counter

import nltk
import numpy as np
from sklearn.preprocessing import FunctionTransformer

from tklearn.utils.collections import dmerge
from tklearn.utils.resource import resource_path

__all__ = ['LexiconVectorizer']


# ======================================================================================================================
# Lexicon Featurizer----------------------------------------------------------------------------------------------------

class SentiWordnetScorer:
    def __init__(self, lexicon_path):
        self.lexicon_path = lexicon_path
        self.lexicon_map_ = self.load_lexicon()

    def transform(self, text):
        """
        This function returns sum of intensities of positive and negative tokens
        :param text: text to featurize
        :param tokenizer: tokenizer to tokenize text
        """
        _word_pattern = re.compile('\w+', flags=re.UNICODE)
        tokens = _word_pattern.findall(text)
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

    def transform(self, text):
        """
        This function returns count of positive and negative tokens
        :param tokens: tokens to featurize
        :param tokenizer: tokenizer to tokenize text
        """
        _word_pattern = re.compile('\w+', flags=re.UNICODE)
        tokens = _word_pattern.findall(text)
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
        return dmerge(*temp)


class PolarityScorer:
    """
    Gives the input tokens a polarity score based on lexicons
    """

    def __init__(self, *lexicon_paths, **kwargs):
        self.bigram = kwargs['bigram'] if 'bigram' in kwargs else True  # Default value is False
        lexicon_maps = []
        for lexicon_path in lexicon_paths:
            with gzip.open(lexicon_path, 'rb') as f:
                lines = f.read().decode('utf8').splitlines()
                lexicon_map = {}
                for l in lines:
                    splits = l.split('\t')
                    lexicon_map[splits[0]] = float(splits[1])
                lexicon_maps.append(lexicon_map)
        self.lexicon_map_ = dmerge(*lexicon_maps)

    def transform(self, text):
        re_str = "[:)(;=\\<8Xx][\^:)(?>|\[\]{}/\\*\-DPSp'3XxOo]+|♥|#?[\w-]+"
        _word_pattern = re.compile(re_str, flags=re.UNICODE)
        tokens = _word_pattern.findall(text)
        unigrams = tokens
        if self.bigram:
            bt = nltk.bigrams(tokens)
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
            token = ' '.join(token) if isinstance(token, tuple) else token
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

    def transform(self, text):
        _word_pattern = re.compile('\w+', flags=re.UNICODE)
        tokens = _word_pattern.findall(text)
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

    def transform(self, text):
        count = 0
        _word_pattern = re.compile('\w+', flags=re.UNICODE)
        tokens = _word_pattern.findall(text)
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

    def transform(self, text):
        """This function returns score of tokens belonging to different emotions
        :param text: text to featurize
        :param tokenizer: tokenizer to tokenize text
        """
        _word_pattern = re.compile('\w+', flags=re.UNICODE)
        tokens = _word_pattern.findall(text)
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
            os.environ['CLASSPATH'] += os.pathsep + jar_path
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

    def transform(self, text):
        """This function returns sum of intensities of positive and negative tokens
        :param text text to featurize
        :param tokenizer tokenizer to tokenize text
        """
        if not self.load_success:
            return [0.0, 0.0]
        _word_pattern = re.compile('\w+', flags=re.UNICODE)
        tokens = _word_pattern.findall(text)
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

    def transform(self, text):
        """
        This function returns count of positive and negative tokens
        :param tokens: tokens to featurize
        """
        liwc = {}

        _word_pattern = re.compile('\w+', flags=re.UNICODE)
        tokens = _word_pattern.findall(text)
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


# --------------------------------------------------------

class LexiconVectorizer(FunctionTransformer):
    """
    Convert a collection of raw documents to a vector of features
    based on the features provided as in `AffectiveTweets` package
    available at [AffectiveTweets](https://github.com/felipebravom/AffectiveTweets).
    """

    __mem_cache = defaultdict(dict)

    def __init__(self, filters=None, caching=None):
        """
        Initialize :AffectiveTweetsVectorizer object.

        :param filters: list of names of filters to be used
        :param caching: whether to use mem cache or not
        """
        super(LexiconVectorizer, self).__init__(self._get_features, validate=False)
        self.filters = filters
        self.caching = caching
        self.initialize()

    def initialize(self):
        if isinstance(self.filters, str):
            self.filters = [self.filters]

    def _get_features(self, seq):
        features_lst = []
        for f in self.filters:
            features = self._extract_features(seq, f)
            if features_lst is not None:
                features_lst += [features]
        return np.concatenate(features_lst, axis=1)

    def _extract_features(self, seq, feature):
        outs = []
        extract_features = LexiconVectorizer._get_filter(feature)
        if extract_features is None:
            return None
        for text in seq:
            if self.caching and text in LexiconVectorizer.__mem_cache[feature]:
                temp = LexiconVectorizer.__mem_cache[feature][text]
            else:
                temp = extract_features.transform(text)
            outs.append(np.array(temp))
            if self.caching and text not in LexiconVectorizer.__mem_cache[feature]:
                LexiconVectorizer.__mem_cache[feature][text] = temp
        outs = np.array(outs)
        if len(outs.shape) == 1:
            outs = np.reshape(outs, (-1, 1))
        return outs

    @staticmethod
    def has_filter(filter):
        featurizers = ['MPQA', 'BingLiu', 'AFINN', 'Emoticon', 'Sentiment140', 'NRCHS', 'NRCWEA', 'NRCHEA', 'NRCAI',
                       'NRC10E',
                       'SentiWordNet', 'Negations', 'SentiStrength', 'LIWC']
        return filter in featurizers

    @staticmethod
    def _get_filter(f):
        f = f.lower()
        if f == 'mpqa':
            path = resource_path('lexicons', 'MPQA', 'mpqa.txt.gz')
            return PolarityCounter(path)
        elif f == 'bingliu':
            path = resource_path('lexicons', 'BingLiu', 'BingLiu.txt.gz')
            return PolarityCounter(path)
        elif f == 'afinn':
            path = resource_path('lexicons', 'AFINN', 'AFINN-en-165.txt.gz')
            return PolarityScorer(path)
        elif f == 'emoticon':
            path = resource_path('lexicons', 'Emoticon', 'AFINN-emoticon-8.txt.gz')
            return PolarityScorer(path)
        elif f == 'sentiment140':
            path = resource_path('lexicons', 'Sentiment140', 'unigrams-pmilexicon.txt.gz')
            path2 = resource_path('lexicons', 'Sentiment140', 'bigrams-pmilexicon.txt.gz')
            return PolarityScorer(path, path2)
        elif f == 'nrchs':
            path = resource_path('lexicons', 'NRCHS', 'unigrams-pmilexicon.txt.gz')
            path2 = resource_path('lexicons', 'NRCHS', 'bigrams-pmilexicon.txt.gz')
            return PolarityScorer(path, path2)
        elif f == 'nrcwea':
            path = resource_path('lexicons', 'NRCWEA', 'NRC-emotion-lexicon-wordlevel-v0.92.txt.gz')
            return EmotionLexiconScorer(path)
        elif f == 'nrchea':
            path = resource_path('lexicons', 'NRCHEA', 'NRC-Hashtag-Emotion-Lexicon-v0.2.txt.gz')
            return EmotionLexiconScorer(path)
        elif f == 'nrcai':
            path = resource_path('lexicons', 'NRCAI', 'nrc_affect_intensity.txt.gz')
            return EmotionLexiconScorer(path)
        elif f == 'nrc10e':
            path = resource_path('lexicons', 'NRC10E', 'w2v-dp-BCC-Lex.txt.gz')
            return EmotionLexiconScorer(path)
        elif f == 'sentiwordnet':
            path = resource_path('lexicons', 'SentiWordNet', 'SentiWordNet_3.0.0.txt.gz')
            return SentiWordnetScorer(path)
        elif f == 'negations':
            path = resource_path('lexicons', 'Negations', 'NegatingWordList.txt.gz')
            return NegationCounter(path)
        elif f == 'sentistrength':
            path = resource_path('lexicons', 'SentiStrength', 'SentiStrength.jar')
            path2 = resource_path('lexicons', 'SentiStrength', 'SentiStrength', '')
            return SentiStrengthScorer(path, path2)
        elif f == 'liwc':
            path = resource_path('lexicons', 'LIWC', 'LIWC2007.dic')
            return LIWCExtractor(path)
        else:
            return None
