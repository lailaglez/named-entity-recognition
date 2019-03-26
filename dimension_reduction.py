from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

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

for name in os.listdir('result'):
    with open(os.path.join('result', name)) as f:
        features.extend(json.load(f))

X = np.array([{k: v for k, v in d.items() if k != 't.ner'} for d in features])
y = np.array([d['t.ner'][:-1] for d in features])

print('Vectorization')

X = DictVectorizer().fit_transform(X, y)

print('Reduction')

svd = TruncatedSVD()
X = svd.fit_transform(X, y)

print()