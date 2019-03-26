from sklearn_crfsuite import CRF, metrics
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from plotting import plot_confusion_matrix
from hmmlearn.hmm import GaussianHMM
from seqlearn.hmm import MultinomialHMM
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import warnings
import json
import os

warnings.filterwarnings('ignore')

tweets = []

for name in os.listdir('conll-ultimate'):
    with open(os.path.join('conll-ultimate', name)) as f:
        tweets.append(json.load(f))

X = []
y = []

for t in tweets:
    X.append([])
    y.append([w['t.ner'] for w in t])
    for w in t:
        X[-1].append({k: v for k, v in w.items() if k != 't.ner'})

X = np.array(X)
y = np.array(y)

classes = np.unique(y)

kf = KFold(n_splits=4)
kf.get_n_splits(X, y)

classifiers = [
    # CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=10, all_possible_transitions=True),
    # CRF(algorithm='l2sgd', c2=0.1, all_possible_transitions=True),
    # CRF(algorithm='l2sgd', c2=0.1),
    CRF(algorithm='lbfgs', all_possible_transitions=True),
    # CRF(algorithm='l2sgd', all_possible_transitions=True),
    # CRF(algorithm='ap', all_possible_transitions=True),
    # CRF(algorithm='pa', all_possible_transitions=True),
    # CRF(algorithm='arow', all_possible_transitions=True),
    # CRF(algorithm='lbfgs', max_iterations=1000, all_possible_transitions=True),
]
classifier_names = [
    "CRF (LBFGS)",
    # "CRF (L2SGD)",
    # "CRF (AP)",
    # "CRF (PA)",
    # "CRF (AROW)",
]


all_labels = list(np.unique([i for a in y for i in a ]))
labels = list(np.unique([i for a in y for i in a ]))
labels.remove('O')
labels = sorted(labels)

for i, classifier in enumerate(classifiers):
    print(classifier_names[i])
    cnf_matrices = []
    recall_scores = []
    precision_scores = []
    f1_scores = []
    recall_scores_only_entities = []
    precision_scores_only_entities = []
    f1_scores_only_entities = []

    for j in range(5):
        print(j)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)

            # print(metrics.flat_classification_report(
            #     y_test, y_pred, labels=labels, digits=3
            # ))

            y_test = [i for a in y_test for i in a]
            y_pred = [i for a in y_pred for i in a]

            cnf_matrices.append(confusion_matrix(y_test, y_pred, labels=all_labels))

            recall_scores.append(recall_score(y_test, y_pred, average='weighted', labels=all_labels))
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', labels=all_labels))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', labels=all_labels))

            recall_scores_only_entities.append(recall_score(y_test, y_pred, labels=labels, average='weighted'))
            precision_scores_only_entities.append(precision_score(y_test, y_pred, labels=labels, average='weighted'))
            f1_scores_only_entities.append(f1_score(y_test, y_pred, labels=labels, average='weighted'))

    info = classifier.tagger_.info()

    # def print_transitions(trans_features):
    #     for (label_from, label_to), weight in trans_features:
    #         print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
    #
    # print("\nTop likely transitions:")
    # print_transitions(Counter(info.transitions).most_common()[:15])
    #
    # print("\nTop unlikely transitions:")
    # unlikely = Counter(info.transitions).most_common()[-15::-1]
    # print_transitions(unlikely)
    #
    # def print_state_features(state_features):
    #     for (attr, label), weight in state_features:
    #         print("%0.6f %-6s %s" % (weight, label, attr))
    #
    # print("\nTop positive:")
    # print_state_features(Counter(info.state_features).most_common()[:15])
    #
    # print("\nTop negative:")
    # unlikely = Counter(info.state_features).most_common()[-15::-1]
    # unlikely.reverse()
    # print_state_features(unlikely)

    def print_transitions(trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    print("\nTop likely transitions:")
    print_transitions(Counter(info.transitions).most_common(15))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(info.transitions).most_common()[-15:])

    def print_state_features(state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-6s %s" % (weight, label, attr))

    print("\nTop positive:")
    print_state_features(Counter(info.state_features).most_common(15))

    print("\nTop negative:")
    print_state_features(Counter(info.state_features).most_common()[-15:])


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

    # # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=all_labels, title=classifier_names[i])
    # # Plot normalized confusion matrix
    # # plt.figure()
    # # plot_confusion_matrix(cnf_matrix, classes=all_labels, normalize=True, title=classifier_names[i])
    plt.show()
    #
