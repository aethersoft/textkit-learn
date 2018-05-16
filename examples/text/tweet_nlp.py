from pprint import pprint

from tklearn.text.twitter.TwitterNLP import TweetNLP

TWEET_NLP_PATH = 'D:\\Programs\\TweetNLP\\ark-tweet-nlp-0.3.2.jar'


def test_tweet_tag():
    tweets = ['this is a message', 'and a second message']
    tweet_nlp = TweetNLP(TWEET_NLP_PATH)
    parsed_results = tweet_nlp.tag(tweets)
    pprint(parsed_results)


def test_get_cluster():
    tweet_nlp = TweetNLP(TWEET_NLP_PATH)
    print(tweet_nlp.get_cluster('I'))


def test_get_cluster_size():
    tweet_nlp = TweetNLP(TWEET_NLP_PATH)
    print(tweet_nlp.cluster_size())


def test_get_cluster_labels():
    tweet_nlp = TweetNLP(TWEET_NLP_PATH)
    print(tweet_nlp.get_cluster('I', output=0))


if __name__ == '__main__':
    test_get_cluster_size()
    test_get_cluster()
    test_get_cluster_labels()
