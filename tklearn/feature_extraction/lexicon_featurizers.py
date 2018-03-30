import csv
import gzip
import os
from collections import defaultdict

from tklearn.utils import bigrams, merge_dicts


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

    def __init__(self, lexicons_path, id):
        rows = csv.DictReader(open(lexicons_path, 'r', encoding="utf8"), )
        emoji_map = {}
        for row in rows:
            emoji_map[row['Emoji']] = [
                float(row['Negative']) / float(row['Occurrences']),
                float(row['Positive']) / float(row['Occurrences']),
                float(row['Neutral']) / float(row['Occurrences'])
            ]
        self.emoji_map = emoji_map
        self.id = id
        self.features = [self.id + '_' + x for x in ['Negative', 'Neutral', 'Positive']]

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
