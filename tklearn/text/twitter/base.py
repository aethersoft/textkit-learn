import re
import string

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

FLAGS = re.MULTILINE | re.DOTALL

# Regex pattern declarations
eyes = r'[8:=;]'

nose = r"['`\-]?"

uri_re = r'https?:\/\/\S+\b|www\.(\w+\.)+\S*'

mention_re = r'@\w+'

hashtag_re = r'#\S+'

number_re = r'[-+]?[.\d]*[\d]+[:,.\d]*'

smile_re = r'{}{}[)dD]+|[)dD]+{}{}'.format(eyes, nose, nose, eyes)

lolface_re = r'{}{}p+'.format(eyes, nose)

sadface_re = r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes)

neutralface_re = r"{}{}[\/|l*]".format(eyes, nose)

slash_re = r'/'

heart_re = r'<3'

repeat_re = r"([!?.]){2,}"

elong_re = r'\b(\S*?)(.)\2{2,}(\S*?)\b'

punct_re = '[' + string.punctuation + ']'

allcaps_re = r"([A-Z]){2,}"

non_ascii_re = r'[^\x00-\x7f]'

stops = set(stopwords.words("english"))


def strip_tags(x):
    if x:
        # BeautifulSoup on content
        soup = BeautifulSoup(x, 'html.parser')
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        return soup.get_text()
    else:
        return ''


def strip_stopwords(lst):
    filtered_words = [word for word in lst if word not in stops]
    return ' '.join(filtered_words)


def hashtag_glove(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + re.split(r'(?=[A-Z])', hashtag_body, flags=FLAGS))
    return result


def all_caps_glove(text):
    return text.group().lower() + " <allcaps>"


def hashtag_alt(text):
    text = text.group()
    hashtag_body = text[1:] + '_hashtag'
    return hashtag_body


def all_caps_alt(text):
    return text.group().lower() + '_allcaps'


def elong(text):
    return text.group().lower()

def clean_tweet(text, repl_conf=None):
    if repl_conf is None:
        glove_repl_conf = {
            'uri': '_url',
            'mention': '_user',
            'number': '_number',
            'elong': r'\1\2+\3',
            'hashtag': hashtag_alt,
            'lower': True
        }
        return clean_tweet(text, repl_conf=glove_repl_conf)
    if repl_conf == 'glove':
        glove_repl_conf = {
            'uri': '<url>',
            'mention': '<user>',
            'smile': '<smile>',
            'lolface': '<lolface>',
            'sadface': '<sadface>',
            'neutralface': '<neutralface>',
            'slash': ' / ',
            'heart': '<heart>',
            'number': '<number>',
            'hashtag': hashtag_glove,
            'repeat': r'\1 <repeat>',
            'elong': r'\1\2 <elong>',
            'allcaps': all_caps_glove,
            'lower': True
        }
        return clean_tweet(text, repl_conf=glove_repl_conf)
    elif repl_conf == 'frederic_godin':
        glove_repl_conf = {
            'mention': '_MENTION_',
            'number': '_NUMBER_',
            'uri': '_URL_',
        }
        return clean_tweet(text, repl_conf=glove_repl_conf)
    else:
        # function so code less repetitive
        def re_sub(pattern, repl):
            return re.sub(pattern, repl, text, flags=FLAGS)

        if 'uri' in repl_conf.keys():
            text = re_sub(uri_re, repl_conf['uri'])
        if 'mention' in repl_conf.keys():
            text = re_sub(mention_re, repl_conf['mention'])
        if 'smile' in repl_conf.keys():
            text = re_sub(smile_re, repl_conf['smile'])
        if 'lolface' in repl_conf.keys():
            text = re_sub(lolface_re, repl_conf['lolface'])
        if 'sadface' in repl_conf.keys():
            text = re_sub(sadface_re, repl_conf['sadface'])
        if 'neutralface' in repl_conf.keys():
            text = re_sub(neutralface_re, repl_conf['neutralface'])
        if 'slash' in repl_conf.keys():
            text = re_sub(slash_re, repl_conf['slash'])
        if 'heart' in repl_conf.keys():
            text = re_sub(heart_re, repl_conf['heart'])
        if 'number' in repl_conf.keys():
            text = re_sub(number_re, repl_conf['number'])
        if 'hashtag' in repl_conf.keys():
            text = re_sub(hashtag_re, repl_conf['hashtag'])
        if 'repeat' in repl_conf.keys():
            text = re_sub(repeat_re, repl_conf['repeat'])
        if 'elong' in repl_conf.keys():
            text = re_sub(elong_re, repl_conf['elong'])
        if 'allcaps' in repl_conf.keys():
            text = re_sub(allcaps_re, repl_conf['allcaps'])
        if 'punct' in repl_conf.keys():
            text = re_sub(punct_re, repl_conf['punct'])
        if 'stop_word' in repl_conf.keys() and repl_conf['stop_word'] is False:
            text = strip_stopwords(text.split(' '))
        if 'lower' in repl_conf.keys() and repl_conf['lower'] is True:
            text = text.lower()
        return text
