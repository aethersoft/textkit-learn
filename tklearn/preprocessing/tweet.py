import re
from typing import Text, List

__all__ = ['TweetPreprocessor']


class TweetPreprocessor:
    """Preprocessor for Tweets.

    Instance of this class can be used to create a preprocessor for tour tweet data.
    Several options are provided and you might be using them according to your use case.
    """

    def __init__(self, **kwargs):
        """Initialize `TweetPreprocessor` object."""
        super(TweetPreprocessor, self).__init__()
        self.normalize = (kwargs['normalize'] if 'normalize' in kwargs else []) or []

    @staticmethod
    def get_links(s: Text) -> List[Text]:
        return re.findall(r'(https?://\S+)', s)

    @staticmethod
    def get_mentions(s: Text) -> List[Text]:
        return re.findall("(@[a-zA-Z0-9_]{1,15})", s)

    @staticmethod
    def get_hashtags(s: Text) -> List[Text]:
        return re.findall(r"(#\w+)", s)

    @staticmethod
    def _replace(s: List[Text], old: Text, new: Text) -> List[Text]:
        return [new if x == old else x for x in s if x.strip() != '']

    def preprocess(self, s: Text) -> Text:
        """Preprocess the input text. Expected input is a Tweet text.

        :param s: Input Tweet text.
        :return: Preprocessed tweet.
        """
        if 'link' in self.normalize:
            for link in self.get_links(s):
                s = s.replace(link, '<link>')
        if 'hashtag' in self.normalize:
            for hashtag in self.get_hashtags(s):
                s = s.replace(hashtag, '<hashtag>')
        if 'mention' in self.normalize:
            for mention in self.get_mentions(s):
                s = s.replace(mention, '<mention>')
        tokens = s.split()
        for ns in self.normalize:
            if isinstance(ns, str):
                pass
            elif isinstance(ns, tuple):
                assert len(ns) == 2, \
                    'Required a tuple of size 2 indicating (new_word, old_words) values for the normalization.'
                assert isinstance(ns[1], list), \
                    'Required a list of old values to replace with the new value.'
                for n in ns[1]:
                    tokens = self._replace(tokens, n, ns[0])
        return ' '.join(tokens)
