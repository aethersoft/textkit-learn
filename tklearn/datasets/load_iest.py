import os


def load_iest(path):
    train_path = os.path.join(path, 'train.csv')
    trial_path = os.path.join(path, 'trial.csv')
    trial_label_path = os.path.join(path, 'trial.labels')
    with open(train_path, 'r', encoding='utf-8') as train_file, \
            open(trial_path, 'r', encoding='utf-8') as trial_file, \
            open(trial_label_path, 'r', encoding='utf-8') as trial_label_file:
        train_csv = [t.split('\t') for t in train_file.read().strip().split('\n')]
        trial_csv = [t.split('\t')[1] for t in trial_file.read().strip().split('\n')]
        trial_labels = trial_label_file.read().strip().split('\n')
        assert len(trial_labels) == len(trial_csv), 'Invalid trial file contents.'
        trial_csv = [[l, t] for l, t in zip(trial_labels, trial_csv)]
    return train_csv, trial_csv
