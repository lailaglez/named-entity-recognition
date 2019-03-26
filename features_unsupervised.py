f_brown = open('clusters/brown-clusters').readlines()
f_clark = open('clusters/clark-clusters').readlines()
f_word2vec = open('clusters/word2vec-clusters').readlines()


def features_unsupervised(token):
    features = {}

    for line in f_brown:
        w = line.split('\t')
        if w[1] == token:
            features['brown-7'] = w[0][:7]
            features['brown-9'] = w[0][:9]
            features['brown-11'] = w[0][:11]
            break

    for line in f_clark:
        w = line.split()
        if w[0] == token:
            features['clark'] = w[1]
            break

    for line in f_word2vec:
        w = line.split()
        if w[0] == token:
            features['word2vec'] = w[1]
            break

    return features