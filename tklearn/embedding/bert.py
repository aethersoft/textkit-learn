import torch
from transformers import BertTokenizer, BertModel
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    'BertEmbedding'
]


class BertEmbedding(BaseEstimator, TransformerMixin):
    """Bert is contextual word embedding. This is different from normal word embeddings that it requires entire
    sentence to return embedding for a given word based on the context of that word."""

    def __init__(self, use_mask_token=False):
        """Constructs Bert model."""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.use_mask_token = use_mask_token

    def fit(self, X, y=None):
        """Does nothing and return self.

        Parameters
        ----------
        X
            Input features
        y
            Labels

        Returns
        -------
            returns self.
        """
        return self

    def transform(self, X):
        """Transforms input to word embedding matrix.

        Parameters
        ----------
        X
            Input features

        Returns
        -------
            Returns embedding tensor.
        """
        return self.get_embedding(X)

    def get_embedding(self, x, index=None):
        """Extracts and returns sentence embedding for input sentence.

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
        if self.use_mask_token:
            index = torch.where(tokens == self.tokenizer.mask_token_id)[1]
        if index is not None:
            return output[:, index, :]
        return output
