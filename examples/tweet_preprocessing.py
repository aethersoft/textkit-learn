from tklearn.preprocessing import TweetPreprocessor

# Thanks to [reference](https://www.postplanner.com/blog/13-good-tweets-you-can-totally-copy/) for
#  following tweets and their creators.
tweet_samples = [
    'I wrote this haiku because Twitter has line breaks no other reason',
    'I had a GREAT week, thanks to YOU! If you need anything, please reach out. ❤️ ❤️ ❤️ '
    '#WorldSmileDay pic.twitter.com/ZpVmQPmcyc',
    'My heart goes out to the Malaysian people. This is such a tragedy. Words can\'t express how sad it is. '
    'I wish we could just have peace. #MH17',
    'This is what I\'m in the mood for right now. :) #nom https://t.co/WOE7VAPgci',
    '''My #SocialMedia seniors book in 1200+ stores in Canada, next to #1 best seller by Harper Lee. 
    TY @ShopprsDrugMart pic.twitter.com/gL6WfAVQM1'''
]

if __name__ == '__main__':
    tp = TweetPreprocessor(normalize=['link', 'hashtag', 'mention'])
    preprocessed = tp.fit_transform(tweet_samples)
    print('\n'.join(preprocessed))
