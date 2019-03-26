from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from hmmlearn.hmm import GMMHMM
import numpy as np
import json
import os
import copy

import warnings

import datetime

import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
try:
    from matplotlib.finance import quotes_historical_yahoo_ochl
except ImportError:
    # For Matplotlib prior to 1.5.
    from matplotlib.finance import (
        quotes_historical_yahoo as quotes_historical_yahoo_ochl
    )

warnings.filterwarnings('ignore')

transmat = [[0 for i in range(9)] for j in range(9)]
startprob = [0 for i in range(9)]
not_last_classes = [0 for i in range(9)]

to_number = {'B-LOCATION': 0, 'B-MISC': 1, 'B-ORG': 2, 'B-PERSON': 3,
             'I-LOCATION': 4, 'I-MISC': 5, 'I-ORG': 6, 'I-PERSON': 7, 'O': 8}

features = []
lengths = []

print('Starting')

for name in os.listdir('vectors')[:100]:
    with open(os.path.join('vectors', name)) as f:
        l = json.load(f)
        features.extend(l)
        lengths.append(len(l))
        startprob[to_number[l[0]['t.ner'][:-1]]] += 1
        for i, d in enumerate(l[:-1]):
            current_ner = to_number[d['t.ner'][:-1]]
            next_ner = to_number[l[i+1]['t.ner'][:-1]]
            transmat[current_ner][next_ner] += 1
            not_last_classes[current_ner] += 1

print('Data loaded, transmat done')

s = sum(startprob)
for i in range(9):
    if s > 0:
        startprob[i] /= s

for j,row in enumerate(transmat):
    for i in range(9):
        row[i] /= not_last_classes[j]


transmat_prior = copy.deepcopy(transmat)
startprob_prior = copy.deepcopy(startprob)

for i in range(9):
    startprob_prior[i] += 1

for row in transmat_prior:
    for i in range(9):
        row[i] += 1


X = np.array([{k: v for k, v in d.items() if k == 't.postag' or k == 't'} for d in features])
y = np.array([d['t.ner'][:-1] for d in features])

lengths = lengths[:100]

division = int(0.7*len(lengths))
feature_division = sum(lengths[:division])
feature_end = sum(lengths)

vec = DictVectorizer()
X = vec.fit_transform(X, y)

X_train = X[:feature_division]
y_train = y[:feature_division]
lengths_train = lengths[:division]

X_test = X[feature_division:feature_end]
y_test = y[feature_division:feature_end]
lengths_test = lengths[division:]

# classes, y_classes = np.unique(y_train, return_inverse=True)
#
# end = np.cumsum(lengths_train)
# start = end - lengths_train
#
# first_states = y_classes[start]
#
# init_prob = [list(first_states).count(c) for c in range(9)]
#
# not_last_classes = [x for i,x in enumerate(y_classes) if i not in end]
#
# for j,row in enumerate(transmat):
#     for i in range(9):
#         row[i] /= not_last_classes.count(j)

print('Ready to start')



model = GMMHMM(n_components=9, init_params='mc', startprob_prior=np.array(startprob_prior),
                    transmat_prior=np.array(transmat_prior))
model.transmat_ = np.array(transmat)
model.startprob_ = np.array(startprob)

model.fit(X_train.toarray(), lengths_train)

#
# model = GaussianHMM(n_components=9, init_params='')
# model.fit(X_train.toarray(), lengths_train)
y_pred = model.predict(X_train.toarray())


print(classification_report(list(map(lambda x: to_number[x], y_train)), y_pred))
print()
