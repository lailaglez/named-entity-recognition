import json


upper_f = open('vocabulary/uppercase')
lower_f = open('vocabulary/lowercase')

upper_dic = json.load(upper_f)
lower_dic = json.load(lower_f)


def features_case(tweet):
    features = {}
    features['proper_case_use'] = False

    if all([t.islower() for t in tweet]) or all([t.isupper() for t in tweet]):
        return features

    if tweet[0][0].islower():
        return features

    period_indexes = [i for i, x in enumerate(tweet) if x == '.']
    for i in period_indexes:
        if i + 1 < len(tweet) and tweet[i + 1][0].islower():
            return features

    upper = [x for w in tweet for x in w if x.isupper()]
    if len(upper) / sum([len(w) for w in tweet]) > 0.5:
        return features

    wrong = 0
    for token in tweet:
        upper_freq = upper_dic.get(token, 0)
        lower_freq = lower_dic.get(token, 0)
        total_freq = upper_freq + lower_freq

        if total_freq:
            if upper_freq / total_freq > 0.9 and token[0].islower():
                wrong += 1
            elif lower_freq / total_freq > 0.9 and token[0].isupper():
                wrong += 1

    if wrong / len(tweet) > 0.3:
        return features

    features['proper_case_use'] = True
    return features
