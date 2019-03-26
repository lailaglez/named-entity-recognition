from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron


from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.metrics import classification_report, make_scorer
from sklearn.metrics import recall_score, precision_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from plotting import plot_confusion_matrix

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import make_pipeline

import numpy as np
import matplotlib.pyplot as plt
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

classes = np.unique(y)
classes.sort()
class_count = np.array([((c, np.count_nonzero(y == c))) for c in classes])

classifiers = [
    PassiveAggressiveClassifier(),
    # Perceptron(),
    # DecisionTreeClassifier(),
    # DecisionTreeClassifier(class_weight='balanced'),
    # RandomForestClassifier(),
    # RandomForestClassifier(class_weight='balanced'),
    # AdaBoostClassifier(),
    # MultinomialNB(),
    # SGDClassifier(),
    # SGDClassifier(class_weight='balanced'),
    #
    # # Takes too long,
    # SVC(kernel="linear", C=0.025, class_weight='balanced'),
    # SVC(kernel="linear", C=1, class_weight='balanced'),
    # SVC(gamma=2, C=1, class_weight='balanced'),
    # MLPClassifier(alpha=1),
    #
    # # Too much memory,
    # KNeighborsClassifier(5),

    # Requires dense matrix
    # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    # GradientBoostingClassifier(),
    # QuadraticDiscriminantAnalysis(),
    # GaussianNB()
]

classifier_names = [
    "PassiveAggressiveClassifier()",
    # "Perceptron()"
    # "DecisionTreeClassifier(class_weight='balanced')",
    # "DecisionTreeClassifier()",
    # "RandomForestClassifier(class_weight='balanced')",
    # "RandomForestClassifier()",
    # "AdaBoostClassifier()",
    # "MultinomialNB()",
    # "SGDClassifier(class_weight='balanced')",
    # "SGDClassifier()",
    #
    # "SVC(kernel='linear', C=0.025, class_weight='balanced')",
    # "SVC(kernel='linear', C=1, class_weight='balanced')",
    # "SVC(gamma=2, C=1, class_weight='balanced')",
    # "MLPClassifier(alpha=1)",
    # #
    # "KNeighborsClassifier(n_neighbors=5)",

    # "GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)",
    # "GradientBoostingClassifier()",
    # "QuadraticDiscriminantAnalysis()",
    # "GaussianNB()",
]

skf = StratifiedKFold(n_splits=4)
skf.get_n_splits(X, y)

X = DictVectorizer().fit_transform(X, y)

imb = ADASYN()
X, y = imb.fit(X, y)


for n in [2, 5, 10, 50, 100, 500, 1000]:
    print('SVD:' + str(n))
    X_svd = TruncatedSVD(n_components=n).fit_transform(X, y)
    for i, classifier in enumerate(classifiers):
        # print(classifier_names[i])
        cnf_matrices = []

        recall_scores  = []
        precision_scores = []
        f1_scores = []

        recall_scores_only_entities  = []
        precision_scores_only_entities = []
        f1_scores_only_entities = []

        p_arrays, r_arrays, f1_arrays = [], [], []

        for j in range(5):
            print(j)
            for train_index, test_index in skf.split(X_svd, y):
                X_train, X_test = X_svd[train_index], X_svd[test_index]
                y_train, y_test = y[train_index], y[test_index]

                pipeline = Pipeline([('clf', classifier)])

                # print('Fitting')
                pipeline.fit(X_train, y_train)
                # print('Predicting')
                y_pred = pipeline.predict(X_test)

                cnf_matrices.append(confusion_matrix(y_test, y_pred))

                recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
                precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
                f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

                recall_scores_only_entities.append(recall_score(y_test, y_pred, labels=classes[:-1], average='weighted'))
                precision_scores_only_entities.append(precision_score(y_test, y_pred, labels=classes[:-1], average='weighted'))
                f1_scores_only_entities.append(f1_score(y_test, y_pred, labels=classes[:-1], average='weighted'))

                p, r, f1, s = precision_recall_fscore_support(y_test, y_pred, labels=classes[:-1])
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


        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=classes, title=classifier_names[i])
        # Plot normalized confusion matrix
        # plt.figure()
        # plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True, title=classifier_names[i])
        # plt.show()