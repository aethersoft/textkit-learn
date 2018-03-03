from tklearn.metrics import pearson_corr, spearman_corr

y_true = [0.2, 0.1, 0.5, 0.4]
y_pred = [0.2, 0.1, 0.5, 0.4]

score_pearson = pearson_corr(y_true, y_pred)
score_spearman = spearman_corr(y_true, y_pred)

print("Pearson Score = {}\nSpearman Score = {}".format(score_pearson, score_spearman))
