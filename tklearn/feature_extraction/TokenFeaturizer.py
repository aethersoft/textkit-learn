from collections import defaultdict
from os.path import join

import numpy as np
from sklearn.preprocessing import FunctionTransformer

from tklearn.feature_extraction.embedding_featurizers import ExtractEmbedding
from tklearn.feature_extraction.lexicon_featurizers import PolarityCounter, PolarityScorer, \
    SentimentRanking, NegationCounter, EmotionLexiconScorer, SentiStrengthScorer, SentiWordnetScorer
from tklearn.feature_extraction.linguistic_featurizers import LIWCExtractor
from tklearn.utils import resource_path

# resource paths
afinn_lexicon_path = join(resource_path(), 'AFINN-en-165.txt.gz')
afinn_emoticon_path = join(resource_path(), 'AFINN-emoticon-8.txt.gz')
bing_liu_lexicon_path = join(resource_path(), 'BingLiu.txt.gz')
mpqa_lexicon_path = join(resource_path(), 'mpqa.txt.gz')
nrc_affect_intensity_lexicon_path = join(resource_path(), 'nrc_affect_intensity.txt.gz')
nrc_emotion_lexicon_path = join(resource_path(), 'NRC-emotion-lexicon-wordlevel-v0.92.txt.gz')
nrc_hashtag_sentiment_unigram_lexicon_path = join(resource_path(), 'NRC-Hashtag-Sentiment-Lexicon-v0.1',
                                                  'unigrams-pmilexicon.txt.gz')
nrc_hashtag_sentiment_bigram_lexicon_path = join(resource_path(), 'NRC-Hashtag-Sentiment-Lexicon-v0.1',
                                                 'bigrams-pmilexicon.txt.gz')
sentiment140_unigram_lexicon_path = join(resource_path(), 'Sentiment140-Lexicon-v0.1', 'unigrams-pmilexicon.txt.gz')
sentiment140_bigram_lexicon_path = join(resource_path(), 'Sentiment140-Lexicon-v0.1', 'bigrams-pmilexicon.txt.gz')
nrc_expanded_emotion_lexicon_path = join(resource_path(), 'w2v-dp-BCC-Lex.txt.gz')
nrc_hashtag_emotion_lexicon_path = join(resource_path(), 'NRC-Hashtag-Emotion-Lexicon-v0.2.txt.gz')
senti_strength_jar_path = join(resource_path(), 'SentiStrength.jar')
senti_strength_dir_path = join(resource_path(), 'SentiStrength/')
senti_wordnet_lexicon_path = join(resource_path(), 'SentiWordNet_3.0.0.txt.gz')
negation_lexicon_path = join(resource_path(), 'NegatingWordList.txt.gz')
edinburgh_embedding_path = join(resource_path(), 'w2v.twitter.edinburgh.100d.csv.gz')
emoji_embedding_path = join(resource_path(), 'emoji2vec.txt.gz')
emoint_data = join(resource_path(), 'emoint/')
liwc_lexicon_path = join(resource_path(), 'LIWC2007.dic')
emoji_sentiment_ranking_path = join(resource_path(), 'Emoji_Sentiment_Data_v1.0.csv')

feature_extractors = {
    'mpqa': PolarityCounter(mpqa_lexicon_path),
    'senti_wordnet': SentiWordnetScorer(senti_wordnet_lexicon_path),
    'sentiment140': PolarityScorer(sentiment140_unigram_lexicon_path, sentiment140_bigram_lexicon_path),
    'affinn': PolarityScorer(afinn_lexicon_path, afinn_emoticon_path),
    'bing_liu': PolarityCounter(bing_liu_lexicon_path),
    'emoji_senti_rank': SentimentRanking(emoji_sentiment_ranking_path, 'Emoji_Sentiment_Ranking'),
    'liwc': LIWCExtractor(liwc_lexicon_path),
    'emoji': ExtractEmbedding(emoji_embedding_path),
    'edinburgh': ExtractEmbedding(edinburgh_embedding_path),
    'neg': NegationCounter(negation_lexicon_path),
    'nrc_affect_int': EmotionLexiconScorer(nrc_affect_intensity_lexicon_path),
    'nrc_emotion': EmotionLexiconScorer(nrc_emotion_lexicon_path),
    'nrc_exp_emotion': EmotionLexiconScorer(nrc_expanded_emotion_lexicon_path),
    'nrc_hashtag': EmotionLexiconScorer(nrc_hashtag_emotion_lexicon_path),
    'nrc_hashtag_score': PolarityScorer(nrc_hashtag_sentiment_unigram_lexicon_path,
                                        nrc_hashtag_sentiment_bigram_lexicon_path),
    'senti_streangth': SentiStrengthScorer(senti_strength_jar_path, senti_strength_dir_path)
}


class TokenFeaturizer(FunctionTransformer):
    """
    The token featurizer converts a list of tokenized documents into a vector,
     in the format used for classification.
     The actual feature to be used is provided as parameter
    """

    __mem_cache = defaultdict(dict)

    def __init__(self, features=None, caching=None):
        """
        Creates a LexiconFeaturizer

        :param caching: whether to use in memory cache. useful in cross-validation tasks
        """
        super(TokenFeaturizer, self).__init__(self._get_lexicon_features, validate=False, )
        self.caching = caching
        self.features = features

    def _get_lexicon_features(self, seq):
        """
        Transforms a sequence of token input to a vector based on lexicons.

        :param seq: Sequence of token inputs
        :return: a list of feature vectors extracted from the sequence of texts in the same order
        """
        if isinstance(self.features, str):
            return self._extract_features(seq, self.features)
        else:
            outs = [self._extract_features(seq, f) for f in self.features]
            return np.concatenate(outs, axis=1)

    def _extract_features(self, seq, feature):
        outs = []
        extract_features = feature_extractors[feature]
        for tokens in seq:
            text = ' '.join(tokens)
            if self.caching and text in TokenFeaturizer.__mem_cache[feature]:
                temp = TokenFeaturizer.__mem_cache[feature][text]
            else:
                temp = extract_features(tokens)
            outs.append(np.array(temp))
            if self.caching and text not in TokenFeaturizer.__mem_cache[feature]:
                TokenFeaturizer.__mem_cache[feature][text] = temp
        outs = np.array(outs)
        if len(outs.shape) == 1:
            outs = np.reshape(outs, (-1, 1))
        return outs
