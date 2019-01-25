from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_similarity_score, log_loss

from tklearn.metrics import pearson_corr, spearman_corr


def check_scoring(scoring=None):
    if scoring is None:
        raise ValueError('Please specify the scoring to check.')
    elif scoring == 'accuracy':
        return accuracy_score
    elif scoring == 'precision':
        return precision_score
    elif scoring == 'log_loss':
        return log_loss
    elif scoring == 'precision_micro':
        return lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro')
    elif scoring == 'precision_macro':
        return lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro')
    elif scoring == 'recall':
        return lambda y_true, y_pred: recall_score
    elif scoring == 'recall_micro':
        return lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro')
    elif scoring == 'recall_macro':
        return lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro')
    elif scoring == 'f1':
        return lambda y_true, y_pred: f1_score(y_true, y_pred)
    elif scoring == 'f1_micro':
        return lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro')
    elif scoring == 'f1_macro':
        return lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')
    elif scoring == 'jaccard':
        return lambda y_true, y_pred: jaccard_similarity_score(y_true, y_pred, normalize=True)
    elif scoring == 'pearson_corr':
        return pearson_corr
    elif scoring == 'spearman_corr':
        return spearman_corr
    return lambda y_true, y_pred: 0.0
