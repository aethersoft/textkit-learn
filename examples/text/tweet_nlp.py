from pprint import pprint

from tklearn.text.twitter.TwitterNLP import TweetNLP

tweet_nlp_path = 'D:\\Programs\\TweetNLP\\ark-tweet-nlp-0.3.2.jar'

tweets = ['this is a message', 'and a second message']

tweet_nlp = TweetNLP(tweet_nlp_path)
parsed_results = tweet_nlp.parse(tweets)
pprint(parsed_results)
