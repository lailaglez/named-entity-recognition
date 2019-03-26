from features import token_features
import warnings
import json
import os


warnings.filterwarnings('ignore')

l = os.listdir('data')
l.sort()

for name in l:
    with open(os.path.join('data', name)) as f:
        tokens = []
        if not os.path.exists(os.path.join('ultimate', name)):
            print(name)
            with open(os.path.join('ultimate', name), mode='w+') as f_vectors:
                f.readline()
                tweet = []
                postags = []
                ner = []

                for line in f.readlines():
                    w = line.split('\t')
                    tweet.append(w[0])
                    postags.append(w[1])
                    ner.append(w[2])

                for i, t in enumerate(tweet):
                    tf = token_features(t, postags[i], tweet, postags, i, ner[i])
                    tokens.append(tf)

                json.dump(tokens, f_vectors)

# category_freq = features_global.category_history
#
# for name in os.listdir('vectors'):
#     f = open(os.path.join('vectors', name))
#     l = json.load(f)
#     for d in l:
#         if 'wiki_category' in d:
#             categories = d['wiki_category']
#             most_freq_category = max(categories, key=category_freq.get)
#             d['wiki_category'] = most_freq_category
#
#     f = open(os.path.join('vectors', name), mode='w')
#     json.dump(l, f)
