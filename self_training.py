from sklearn.metrics import recall_score, precision_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from collections import Counter

from plotting import plot_confusion_matrix
from sklearn_crfsuite import CRF
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier
from copy import copy

import matplotlib.pyplot as plt
import warnings
import numpy.random as random
import numpy as np
import json
import os

warnings.filterwarnings('ignore')


def Fleiss_agreement(matrix):
    c = len(matrix)
    s = 0
    for i in range(len(matrix)):
        column = np.array([row[i] for row in matrix])
        labels = np.unique(column)
        s += (1 / (c * (c - 1))) * sum(
            [np.count_nonzero(column == l) * (np.count_nonzero(column == l) - 1) for l in labels])
    return s


def best_prediction(matrix):
    prediction = []
    for i in range(len(matrix)):
        column = np.array([row[i] for row in matrix])
        entity_column = [c for c in column if c != 'O']
        most_common = Counter(entity_column).most_common(1)
        if most_common and most_common[0][1] > 1:
            prediction.append(most_common[0][0])
        else:
            prediction.append('O')




class SelfTrain:
    def __init__(self, n_init, n_step, structured_classifiers, classic_classifiers, data_folder):
        self.n_init = n_init
        self.n_step = n_step
        self.structured_classifiers = structured_classifiers
        self.classic_classifiers = classic_classifiers
        self.data_folder = data_folder

        tweets = []

        for name in os.listdir(data_folder):
            with open(os.path.join(data_folder, name)) as f:
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

        self.structured_X = X
        self.structured_y = y

        features = []
        lengths = []
        for name in os.listdir(data_folder):
            with open(os.path.join(data_folder, name)) as f:
                dic = json.load(f)
                features.extend(dic)
                lengths.append(len(dic))

        X = np.array([{k: v for k, v in d.items() if k == 't.postag'} for d in features])
        y = np.array([d['t.ner'] for d in features])
        lengths = np.array(lengths)

        self.classic_X = X
        self.classic_y = y
        self.lengths = lengths
        self.beginnings = np.array([0] + list(np.cumsum(lengths)))
        self.labels = np.unique(y)

        self.i = 0
        self.ended = False

        self.tweets_for_training = np.array([False] * len(self.structured_y))

        to_train = random.choice([i for i in range(len(self.structured_y)) if not self.tweets_for_training[i]],
                                 self.n_init, replace=False)

        self.tweets_for_training[to_train] = True

        self.tokens_for_training = np.array([False] * sum(self.lengths))
        for i, t in enumerate(self.tweets_for_training):
            if t:
                self.tokens_for_training[range(self.beginnings[i], self.beginnings[i + 1])] = True

        self.original_testing_tokens = np.array(list(map(lambda x: not (x), self.tokens_for_training)))

        self.y_true = copy(self.classic_y[self.original_testing_tokens])


    def divide_data(self):
        tweets_for_training = self.tweets_for_training
        tweets_for_testing = self.tweets_for_testing

        self.structured_train_X = self.structured_X[tweets_for_training]
        self.structured_train_y = self.structured_y[tweets_for_training]

        self.structured_test_X = self.structured_X[tweets_for_testing]
        self.structured_test_y = self.structured_y[tweets_for_testing]

        tokens_for_training = np.array([False] * sum(self.lengths))
        for i, t in enumerate(tweets_for_training):
            if t:
                tokens_for_training[range(self.beginnings[i], self.beginnings[i + 1])] = True

        self.tokens_for_training = tokens_for_training
        tokens_for_testing = np.array(list(map(lambda x: not (x), tokens_for_training)))

        self.classic_train_X = self.classic_X[tokens_for_training]
        self.classic_train_y = self.classic_y[tokens_for_training]

        self.classic_test_X = self.classic_X[tokens_for_testing]
        self.classic_test_y = self.classic_y[tokens_for_testing]

    def step(self):
        if self.ended:
            raise Exception()

        self.divide_data()

        predictions = []

        for classifier in self.structured_classifiers:
            classifier.fit(self.structured_train_X, self.structured_train_y)
            predicted_y = classifier.predict(self.structured_test_X)
            # predicted_y = [i for a in predicted_y for i in a]
            predictions.append(predicted_y)

        for classifier in self.classic_classifiers:
            pipeline = Pipeline([('vect', DictVectorizer()), ('clf', classifier)])
            pipeline.fit(self.classic_train_X, self.classic_train_y)
            predicted_y = pipeline.predict(self.classic_test_X)
            predicted_lengths = self.lengths[self.tweets_for_testing]
            predicted_y = np.split(predicted_y, np.cumsum(predicted_lengths[:-1]))
            predictions.append(predicted_y)

        predictions = np.array(predictions)
        scores = self.agreement(predictions)

        limit = min(self.n_step, len(np.where(self.tweets_for_testing)[0]))

        best_predictions = scores[:limit]
        indexes = np.array([b[2] for b in best_predictions])
        real_indexes = np.where(self.tweets_for_testing)[0]
        training_real_indexes = real_indexes[indexes]
        real_indexes.sort()

        self.tweets_for_training[training_real_indexes] = True

        for p in best_predictions:
            real_index = real_indexes[p[2]]
            prediction = p[1]

            most_common = []

            for i in range(len(prediction[0])):
                most_common.append(Counter([row[i] for row in prediction]).most_common(1)[0][0])

            assert len(most_common) == len(self.structured_y[real_index]) and len(most_common) == self.lengths[
                real_index]

            self.structured_y[real_index] = most_common
            for j in range(len(prediction[0])):
                self.classic_y[self.beginnings[real_index] + j] = most_common[j]

        if len(np.where(self.tweets_for_testing)[0]) == 0:
            self.ended = True

    def agreement(self, predictions):
        scores = []
        for i in range(len(predictions[0])):
            column = predictions[:, i]
            column[0] = np.array(column[0])
            scores.append((Fleiss_agreement(column), column, i))

        s = sorted(scores, reverse=True, key=lambda x: x[0])

        return s

    @property
    def tweets_for_testing(self):
        return np.array(list(map(lambda x: not (x), self.tweets_for_training)))

    @property
    def y_pred(self):
        if not self.ended:
            raise Exception()
        return self.classic_y[self.original_testing_tokens]


cnf_matrices = []

recall_scores = []
precision_scores = []
f1_scores = []

recall_scores_only_entities = []
precision_scores_only_entities = []
f1_scores_only_entities = []

p_arrays, r_arrays, f1_arrays = [], [], []

for i in range(5):
    print('Initiating trainer')
    trainer = SelfTrain(n_init=500, n_step=500, structured_classifiers=[CRF()],
                        classic_classifiers=[Perceptron(), PassiveAggressiveClassifier()],
                        data_folder='data')
    j = 0
    while not trainer.ended:
        print('Beginning step ' + str(j))
        trainer.step()
        j += 1

    y_true = trainer.y_true
    y_pred = trainer.y_pred
    classes = sorted(trainer.labels)

    cnf_matrices.append(confusion_matrix(y_true, y_pred))

    recall_scores.append(recall_score(y_true, y_pred, average='weighted'))
    precision_scores.append(precision_score(y_true, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_true, y_pred, average='weighted'))

    recall_scores_only_entities.append(
        recall_score(y_true, y_pred, labels=classes[:-1], average='weighted'))
    precision_scores_only_entities.append(
        precision_score(y_true, y_pred, labels=classes[:-1], average='weighted'))
    f1_scores_only_entities.append(f1_score(y_true, y_pred, labels=classes[:-1], average='weighted'))

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=classes)
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
plot_confusion_matrix(cnf_matrix, classes=classes, title='Self-training')
# Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True, title=classifier_names[i])
plt.show()
