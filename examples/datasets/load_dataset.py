from tklearn.datasets import load_emoint
import pandas as pd

print('Loading data...', end=' ')
train, dev, test = load_emoint(input_path='D:\\Documents\\Resources\\Datasets\\EmoInt-2017\\')
print('[Done]')

print('Shape of Training Set={}\nShape of Dev Set={}\n{}'.format(train.shape, dev.shape, '-' * 25))

# try merging datasets
frames = [train, dev]

result = pd.concat(frames)
print('Shape of combined dataset = {}'.format(result.shape))
assert result.shape[0] == dev.shape[0] + train.shape[0], 'Combining Training and Dev datasets failed.'

train_anger = result[result.emotion == 'anger']

print(train_anger[:5])
