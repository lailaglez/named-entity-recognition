from sklearn.metrics import classification_report
from nltk.classify import maxent
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from plotting import plot_confusion_matrix


import matplotlib.pyplot as plt
import numpy as np
import warnings
import json
import os

warnings.filterwarnings('ignore')


features = []

for name in os.listdir('data'):
    with open(os.path.join('data', name)) as f:
        features.extend(json.load(f))

X = np.array([{k: v for k, v in d.items() if k != 't.ner'} for d in features])
y = np.array([d['t.ner'] for d in features])

all_labels = list(np.unique(y))
labels = list(np.unique(y))
labels.remove('O')
labels = sorted(labels)

divisions = [0, int(0.25*len(y)), int(0.5*len(y)), int(0.75*len(y)), len(y)]

cnf_matrices = []
recall_scores  = []
precision_scores = []
f1_scores = []
recall_scores_only_entities = []
precision_scores_only_entities = []
f1_scores_only_entities = []

for i in range(5):
    print(i)
    for j in range(1,len(divisions)):
        test_indices = np.arange(divisions[j-1], divisions[j])
        train_indices = np.concatenate((np.arange(0, divisions[j - 1]), np.arange(divisions[j], len(y))))

        X_train = X[train_indices]
        y_train = y[train_indices]
        data_train = list(zip(X_train,y_train))

        X_test = X[test_indices]
        y_test = y[test_indices]

        encoding = maxent.TypedMaxentFeatureEncoding.train(data_train, count_cutoff=3, alwayson_features=True)
        classifier = maxent.MaxentClassifier.train(data_train, bernoulli=False, encoding=encoding, trace=0)
        y_pred = classifier.classify_many(X_test)

        cnf_matrices.append(confusion_matrix(y_test, y_pred, labels=all_labels))

        recall_scores.append(recall_score(y_test, y_pred, average='weighted', labels=all_labels))
        precision_scores.append(precision_score(y_test, y_pred, average='weighted', labels=all_labels))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted', labels=all_labels))

        recall_scores_only_entities.append(recall_score(y_test, y_pred, labels=labels, average='weighted'))
        precision_scores_only_entities.append(precision_score(y_test, y_pred, labels=labels, average='weighted'))
        f1_scores_only_entities.append(f1_score(y_test, y_pred, labels=labels, average='weighted'))

cnf_matrix = np.round(np.mean(cnf_matrices, axis=0))

recall = np.mean(recall_scores)
precision = np.mean(precision_scores)
f1 = np.mean(f1_scores)

recall_only_entities = np.mean(recall_scores_only_entities)
precision_only_entities = np.mean(precision_scores_only_entities)
f1_only_entities = np.mean(f1_scores_only_entities)

print('MaxEntropy()')
# print(cnf_matrix)
print('Precision: ' + str(precision))
print('Recall: ' + str(recall))
print('F1: ' + str(f1))
print('Precision (only entities): ' + str(precision_only_entities))
print('Recall (only entities): ' + str(recall_only_entities))
print('F1 (only entities): ' + str(f1_only_entities))
print()

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=all_labels, title='MaxEntropy()')
# Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=all_labels, normalize=True, title=classifier_names[i])
plt.show()