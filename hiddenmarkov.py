from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from plotting import plot_confusion_matrix
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import GMMHMM
from seqlearn.hmm import MultinomialHMM
from sklearn.decomposition import PCA, TruncatedSVD


import matplotlib.pyplot as plt
import numpy as np
import warnings
import json
import os

warnings.filterwarnings('ignore')


features = []
lengths = []

for name in os.listdir('ultimate'):
    with open(os.path.join('ultimate', name)) as f:
        dic = json.load(f)
        features.extend(dic)
        lengths.append(len(dic))

X = np.array([{k: v for k, v in d.items() if k != 't.ner'} for d in features])
y = np.array([d['t.ner'] for d in features])
lengths = np.array(lengths)

labels, y = np.unique(y, return_inverse=True)

divisions = [0, int(0.25*len(lengths)), int(0.5*len(lengths)), int(0.75*len(lengths)), len(lengths)]
feature_divisions = [sum(lengths[:division]) for division in divisions]

classifiers = [
    MultinomialHMM(),
    # MultinomialHMM(alpha=1),
    # MultinomialHMM(alpha=0.00001),
    # MultinomialHMM(alpha=10),
    # MultinomialHMM(decode='bestfirst'),
    # GaussianHMM(n_components=9),
    # GaussianHMM(n_components=9, algorithm='map'),
    # GMMHMM(n_components=9),
    # GMMHMM(n_components=9, algorithm='map'),
]
classifier_names = [
    "MultinomialHMM()",
    # "MultinomialHMM(alpha=1)",
    # "MultinomialHMM(alpha=0.00001)",
    # "MultinomialHMM(alpha=10)",
    # "MultinomialHMM(decode='bestfirst')",
    # "GaussianHMM()",
    # "GaussianHMM(algorithm='map')",
    # "GMMHMM()",
    # "GMMHMM(algorithm='map')"
]

vec = DictVectorizer()
X = vec.fit_transform(X, y)

X = TruncatedSVD(n_components=500).fit_transform(X, y)

for i, classifier in enumerate(classifiers):
    print(classifier_names[i])

    cnf_matrices = []
    recall_scores  = []
    precision_scores = []
    f1_scores = []
    recall_scores_only_entities = []
    precision_scores_only_entities = []
    f1_scores_only_entities = []

    for k in range(30):
        print(k)
        for j in range(1,len(divisions)):
            test_index_features = np.arange(feature_divisions[j-1], feature_divisions[j])
            test_indices_lengths = np.arange(divisions[j-1], divisions[j])

            train_indices_features = np.concatenate((np.arange(0, feature_divisions[j - 1]), np.arange(feature_divisions[j], len(y))))
            train_indices_lengths = np.concatenate((np.arange(0, divisions[j - 1]), np.arange(divisions[j], len(lengths))))

            X_train = X[train_indices_features]
            y_train = y[train_indices_features]
            lengths_train = lengths[train_indices_lengths]

            X_test = X[test_index_features]
            y_test = y[test_index_features]
            lengths_test = lengths[test_indices_lengths]

            classifier.fit(X_train, y_train, lengths_train)
            y_pred = classifier.predict(X_test)

            cnf_matrices.append(confusion_matrix(y_test, y_pred, labels=range(9)))

            recall_scores.append(recall_score(y_test, y_pred, average='weighted', labels=range(9)))
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', labels=range(9)))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', labels=range(9)))

            recall_scores_only_entities.append(recall_score(y_test, y_pred, labels=range(8), average='weighted'))
            precision_scores_only_entities.append(precision_score(y_test, y_pred, labels=range(8), average='weighted'))
            f1_scores_only_entities.append(f1_score(y_test, y_pred, labels=range(8), average='weighted'))

    cnf_matrix = np.round(np.mean(cnf_matrices, axis=0))

    recall = np.mean(recall_scores)
    precision = np.mean(precision_scores)
    f1 = np.mean(f1_scores)

    recall_only_entities = np.mean(recall_scores_only_entities)
    precision_only_entities = np.mean(precision_scores_only_entities)
    f1_only_entities = np.mean(f1_scores_only_entities)

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
    plot_confusion_matrix(cnf_matrix, classes=range(9), title=classifier_names[i])
    # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=range(9), normalize=True, title=classifier_names[i])
    # plt.show()
