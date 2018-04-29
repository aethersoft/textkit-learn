import os
import shlex
from subprocess import run, PIPE
from tempfile import mkstemp


class TweetNLP:
    TAGGER_CMD = "java -XX:ParallelGCThreads=2 -Xmx500m -jar "

    def __init__(self, tweet_nlp_path):
        self.tagger_cmd = TweetNLP.TAGGER_CMD + tweet_nlp_path.replace('\\', '\\\\')

    def parse(self, tweets):
        """
        Parse a list of tweets

        :param tweets: a list of tweets
        :return: parsed results of tweets provided
        """
        output, error = self._call_run_tagger(tweets)
        if self._check_error(error):
            rows = output.strip('\n\r').split('\n\r\n')
            rows = [self._split_results(row) for row in rows]
            return rows
        else:
            return None

    def _call_run_tagger(self, tweets):
        """
        Run tagger on input tweets

        :param tweets: input tweets
        :return: tagger results
        """

        # remove carriage returns as they are tweet separators for the stdin interface
        tweets_cleaned = [tw.replace('\n', ' ') for tw in tweets]
        message = "\n".join(tweets_cleaned)

        fd, tmp_path = mkstemp()
        try:
            with os.fdopen(fd, 'w') as tmp:
                tmp.write(message)
            # cmd = self.tagger_cmd + ' ' + tmp_path
            args = shlex.split(self.tagger_cmd)
            args.append('--output-format')
            args.append('conll')
            args.append(tmp_path)
            print(args)
            p = run(args, stdout=PIPE, stderr=PIPE)
            error = p.stderr
            output = p.stdout
        finally:
            os.remove(tmp_path)
        assert not os.path.exists(tmp_path), 'Unable to delete temporary file.'
        return output.decode('utf-8'), error.decode('utf-8')

    @staticmethod
    def _check_error(error):
        error = error.split('\n')
        if len(error) >= 2 and error[1].strip().startswith('Tokenized and tagged'):
            return True
        else:
            return False

    @staticmethod
    def _split_results(lines):
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
