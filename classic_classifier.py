from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import classification_report, make_scorer
from sklearn.metrics import cohen_kappa_score, average_precision_score, precision_recall_curve, recall_score, \
    precision_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from plotting import plot_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
import inspect
import json
import os

warnings.filterwarnings('ignore')

features = []

for name in os.listdir('data'):
    with open(os.path.join('data', name)) as f:
        features.extend(json.load(f))

X = np.array([{k: v for k, v in d.items() if k!='t.ner'} for d in features])
y = np.array([d['t.ner'] for d in features])

classes = np.unique(y)
class_count = np.array([((c, np.count_nonzero(y == c))) for c in classes])

classifiers = [
    # DummyClassifier(),
    # DecisionTreeClassifier(class_weight='balanced'),
    # SGDClassifier(),
    # Perceptron(),
    PassiveAggressiveClassifier(),
    # AdaBoostClassifier(),
    # RandomForestClassifier(),
    # SVC(),
    # MultinomialNB(),
    # DecisionTreeClassifier(class_weight='balanced'),
    # RandomForestClassifier(),
    # RandomForestClassifier(class_weight='balanced'),
    # AdaBoostClassifier(),
    # PassiveAggressiveClassifier(),
    # Perceptron(),
    # ExtraTreesClassifier(),
    # Perceptron(class_weight='balanced'),
    # SGDClassifier(shuffle=True),
    # SGDClassifier(class_weight='balanced', shuffle=True),
    # MultinomialNB(),
    # BernoulliNB(),

    # # Takes too long,
    # SVC(kernel="linear", C=0.025, class_weight='balanced'),
    # SVC(kernel="linear", C=1, class_weight='balanced'),
    # SVC(gamma=2, C=1, class_weight='balanced'),
    # MLPClassifier(alpha=1),
    #
    # # Too much memory,
    # KNeighborsClassifier(2),
    #
    # # Requires dense matrix
    # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    # GradientBoostingClassifier(),
    # QuadraticDiscriminantAnalysis(),
    # LinearDiscriminantAnalysis(),
    # GaussianNB()
]

classifier_names = [
    # "DummyClassifier()",
    # "DecisionTreeClassifier()",
    # "SGDClassifier()",
    # "Perceptron()",
    "PassiveAggressiveClassifier()",
    # "AdaBoostClassifier()",
    # "RandomForestClassifier()",
    # "SVC()",
    # "MultinomialNB()",
    # "DummyClassifier()",
    # "DecisionTreeClassifier(class_weight='balanced')",
    # "DecisionTreeClassifier()",
    # "RandomForestClassifier(class_weight='balanced')",
    # "RandomForestClassifier()",
    # "AdaBoostClassifier()",
    # "PassiveAggressiveClassifier()",
    # "Perceptron()",
    # "ExtraTreesClassifier()",
    # "Perceptron(class_weight='balanced')",
    # "SGDClassifier(class_weight='balanced', shuffle=True)",
    # "SGDClassifier(shuffle=True)",
    # "MultinomialNB()",
    # "BernoulliNB()",
    # "SVC(kernel='linear', C=0.025, class_weight='balanced')",
    # "SVC(kernel='linear', C=1, class_weight='balanced')",
    # "SVC(gamma=2, C=1, class_weight='balanced')",
    # "MLPClassifier(alpha=1)",
    #
    # "KNeighborsClassifier(n_neighbors=5)",
    #
    # "GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)",
    # "GradientBoostingClassifier()",
    # "LinearDiscriminantAnalysis()",
    # "QuadraticDiscriminantAnalysis()",
    # "GaussianNB()",
]

# ensemble_classifiers = [VotingClassifier(estimators=list(zip(classifier_names, classifiers)), voting='soft'),
# VotingClassifier(estimators=list(zip(classifier_names, classifiers)), voting='hard')
# ]

# ensemble_classifier_names = ["VotingClassifier(voting='soft')",
# "VotingClassifier(voting='hard')",
# ]

skf = StratifiedKFold(n_splits=4)
skf.get_n_splits(X, y)

for i, classifier in enumerate(classifiers):
    print(classifier_names[i])
    cnf_matrices = []

    recall_scores = []
    precision_scores = []
    f1_scores = []

    recall_scores_only_entities = []
    precision_scores_only_entities = []
    f1_scores_only_entities = []

    p_arrays, r_arrays, f1_arrays = [], [], []

    for j in range(15):
        print(j)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            pipeline = Pipeline([('vect', DictVectorizer()), ('clf', classifier)])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            cnf_matrices.append(confusion_matrix(y_test, y_pred))

            recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
            precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

            recall_scores_only_entities.append(
                recall_score(y_test, y_pred, labels=classes[:-1], average='weighted'))
            precision_scores_only_entities.append(
                precision_score(y_test, y_pred, labels=classes[:-1], average='weighted'))
            f1_scores_only_entities.append(f1_score(y_test, y_pred, labels=classes[:-1], average='weighted'))

            p, r, f1, s = precision_recall_fscore_support(y_test, y_pred, labels=classes)
            p_arrays.append(p)
            r_arrays.append(r)
            f1_arrays.append(f1)

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

    print(classes)

    p_by_class = np.array(p_arrays).mean(axis=0)
    print('Precision by class: ' + str(p_by_class))
    print('Mean: ' + str(np.mean(p_by_class)))

    r_by_class = np.array(r_arrays).mean(axis=0)
    print('Recall by class: ' + str(r_by_class))
    print('Mean: ' + str(np.mean(r_by_class)))

    f1_by_class = np.array(f1_arrays).mean(axis=0)
    print('F1 by class: ' + str(f1_by_class))
    print('Mean: ' + str(np.mean(f1_by_class)))

    print()

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes, title=classifier_names[i])
    # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True, title=classifier_names[i])
    # plt.show()
