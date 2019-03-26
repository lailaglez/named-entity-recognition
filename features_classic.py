import re


def features_classic(token, pre_token, suc_token, postag=None, pre_postag=None, suc_postag=None):
    features = {}

    # token
    features['t'] = token
    features['t.lower'] = token.lower()
    features['t.isupper'] = token.isupper()
    features['t.istitle'] = token.istitle()
    features['t.isdigit'] = token.isdigit()
    features['t.shape'] = shape(token)
    features['t.postag'] = postag

    # context
    if pre_token:
        features['t-1'] = pre_token
        features['t-1.shape'] = shape(pre_token)
        features['t-1.postag'] = pre_postag
    else:
        features['t.first'] = True
    if suc_token:
        features['t1'] = suc_token
        features['t1.shape'] = shape(suc_token)
        features['t1.postag'] = suc_postag
    else:
        features['t.last'] = True

    # suffix and prefix
    features['t[:1]'] = token[:1]
    features['t[:2]'] = token[:2]
    features['t[:3]'] = token[:3]

    features['t[-1:]'] = token[-1:]
    features['t[-2:]'] = token[-2:]
    features['t[-3:]'] = token[-3:]

    return features


def shape(token):
    token = re.sub(r'[a-záéíóúüñ]+', 'a', token)
    token = re.sub(r'[A-ZÁÉÍÓÚÜÑ]+', 'A', token)
    token = re.sub(r'\d+', '0', token)
    token = re.sub(r'[^aA0]+', '-', token)
    return token
