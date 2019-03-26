
from nltk.corpus import stopwords
import json
import os


spanish_stopwords = stopwords.words('spanish')

def filter_stopwords():
    for name in os.listdir('ultimate'):
        print(name)
        with open(os.path.join('ultimate', name)) as f:
            l = json.load(f)
            with(open(os.path.join('sin_stopwords', name), mode='w')) as fd:
                new_l = [d for d in l if d['t.lower'] not in spanish_stopwords and d['t.postag'] != 'PUNCTUATION']
                json.dump(new_l, fd)

# filter_stopwords()