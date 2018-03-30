from tklearn.feature_extraction import LexiconFeaturizer

emoint_features = ['mpqa', 'bing_liu', 'affinn', 'sentiment140', 'nrc_hashtag_score', 'nrc_exp_emotion',
                       'nrc_hashtag', 'senti_wordnet', 'nrc_emotion', 'neg']

f = LexiconFeaturizer(features=emoint_features)

o = f.fit_transform(['this is a good and bad case'.split(' ')])[0]

assert len(o) == 43, 'Invalid feature list'
print('Successfully completed!')