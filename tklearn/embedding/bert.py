from transformers import BertTokenizer, BertModel as _BertModel

__all__ = [
    'BertModel'
]


class BertModel:
    """ Bert is contextual word embedding. This is different from normal word embeddings that it requires entire
    sentence to return embedding for a given word based on the context of that word.
    """

    def __init__(self):
        """ Constructs Bert model."""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = _BertModel.from_pretrained('bert-base-uncased', return_dict=True)

    def predict(self, x, index=None):
        """ Extracts and returns sentence embedding for input sentence.

        Parameters
        ----------
        x
            Iterable of tokenized sentences.
        index
            Index of the word to extract the word embedding. Index starts with one.

        Returns
        -------
            Array of features for each sentences.
        """
        tokens = self.tokenizer(x, return_tensors="pt", is_pretokenized=True, add_special_tokens=True, padding=True,
                                truncation=True, max_length=512)
        output = self.model(**tokens).last_hidden_state.cpu().detach().numpy()
        if index is not None:
            return output[:, index, :]
        return output
