from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from plotting import plot_confusion_matrix

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import NearMiss as balancer
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced

import matplotlib.pyplot as plt
import numpy as np
import json
import os

features = []

for name in os.listdir('vectors'):
    with open(os.path.join('vectors', name)) as f:
        features.extend(json.load(f))

X = np.array([{k: v for k, v in d.items() if k != 't.ner'} for d in features])
y = np.array([d['t.ner'] for d in features])

classes = np.unique(y)
# class_count = np.array([((c, np.count_nonzero(y==c))) for c in classes])

X = DictVectorizer().fit_transform(X[:5000]).toarray()
y = y[:5000]

division = int(0.7*len(X))

train_X = X[:division]
train_y = y[:division]

test_X = X[division:]
test_y = y[division:]

# Create a pipeline
pipeline = make_pipeline(
                         balancer(),
                         LinearSVC())
pipeline.fit(train_X, train_y)
pred_y = pipeline.predict(test_X)

# Classify and report the results
print(classification_report_imbalanced(test_y, pred_y))

#
# pipeline = Pipeline([('vect', DictVectorizer()), ('clf', SVC())])
#
# pipeline.fit(train_X, train_y)
# pred_y = pipeline.predict(test_X)
#
# pred_y_count = np.array([((c, np.count_nonzero(pred_y==c))) for c in classes])
#



print(classification_report(test_y, pred_y))
print(cohen_kappa_score(test_y, pred_y))

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_y, pred_y)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes, title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True, title='Normalized confusion matrix')
plt.show()
