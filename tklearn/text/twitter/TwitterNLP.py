import os
import shlex
from subprocess import run, PIPE
from tempfile import mkstemp


class TweetNLP:
    TAGGER_CMD = 'java -XX:ParallelGCThreads=2 -Xmx500m -jar'
    CLUSTERS_PATH = ['..', 'data', 'wordtypes-v0.1', '50mpaths2.txt']

    def __init__(self, tweet_nlp):
        self.tweet_nlp = tweet_nlp
        self._initialize()

    def _initialize(self):
        f_path = os.path.join(self.tweet_nlp, *TweetNLP.CLUSTERS_PATH)
        self.records_ = {}
        self.labels_ = set()
        with open(f_path, 'r', encoding='utf-8') as f:
            rows = f.readlines()
            for row in rows:
                row = row.strip()
                parts = row.split('\t')
                cluster = [int(i) for i in parts[0]]
                word = parts[1]
                frequency = parts[2]
                index = self._node_index(cluster)
                self.labels_.add(index)
                self.records_[word] = (index, cluster, frequency)

    def tag(self, tweets):
        """
        Parse a list of tweets

        :param tweets: a list of tweets
        :return: parsed results of tweets provided
        """
        output, error = self._call_run_tagger(tweets)
        if self._check_error(error):
            rows = output.strip('\n\r').split('\n\r\n')
            rows = [self._split_tagger_results(row) for row in rows]
            return rows
        else:
            return None

    def get_cluster(self, word, output=1):
        """
        Get cluster

        :param word: input word
        :param output: output type: {'1': cluster tree, '0': cluster labels}
        :return: cluster represented as
        """
        if word in self.records_:
            return self.records_[word][output]
        if output == 0:
            return -1
        return []

    def cluster_size(self):
        return len(self.labels_)

    def _node_index(self, vec):
        index = 0
        for i, k in enumerate(vec):
            index += (2 ** i) * (2 ** k)
        return index

    def _call_run_tagger(self, tweets):
        """
        Run tagger on input tweets

        :param tweets: input tweets
        :return: tagger results
        """
        # remove carriage returns as they are tweet separators for the stdin interface
        tweets_cleaned = [tw.replace('\n', ' ') for tw in tweets]
        message = "\n".join(tweets_cleaned)
        message = message.encode('utf-8').decode('utf-8')

        fd, tmp_path = mkstemp()
        try:
            with open(fd, 'w', encoding='utf-8') as tmp:
                tmp.write(message)
            args = shlex.split(TweetNLP.TAGGER_CMD)
            args.append(self.tweet_nlp.replace('\\', '\\\\'))
            args.append('--output-format')
            args.append('conll')
            args.append(tmp_path)
            p = run(args, stdout=PIPE, stderr=PIPE)
            error = p.stderr
            output = p.stdout
        finally:
            os.remove(tmp_path)
        assert not os.path.exists(tmp_path), 'Unable to delete temporary file.'
        return output.decode('utf-8'), error.decode('utf-8')

    @staticmethod
    def _check_error(error):
        """
        Check for error

        :param error: error output to check for error
        :return: whether there is an error
        """
        error = error.split('\n')
        if len(error) >= 2 and error[1].strip().startswith('Tokenized and tagged'):
            return True
        else:
            return False

    @staticmethod
    def _split_tagger_results(lines):
        """
        Parse the tab-delimited returned lines.

        :param row:
        :return:
        """
        tokens = []
        for line in lines.strip().split('\n'):
            parts = line.strip().split('\t')
            word = parts[0]
            tag = parts[1]
            conf = float(parts[2])
            tokens.append((word, tag, conf))
        return tokens
