from features_classic import features_classic
from features_unsupervised import features_unsupervised
from features_case import features_case
from features_global import features_global


def token_features(token, postag, tweet, tweet_pos, token_index, ner_tag):
    features = {}

    pre_token = tweet[token_index - 1] if token_index > 1 else ''
    pre_pre_token = tweet[token_index - 2] if token_index > 2 else ''
    suc_token = tweet[token_index + 1] if token_index < len(tweet) - 2 else ''
    suc_suc_token = tweet[token_index + 2] if token_index < len(tweet) - 3 else ''
    pre_postag = tweet_pos[token_index - 1] if token_index > 1 else ''
    suc_postag = tweet_pos[token_index + 1] if token_index < len(tweet) - 2 else ''

    features_c = features_classic(token, pre_token, suc_token, postag, pre_postag, suc_postag)
    features.update(features_c)

    features_u = features_unsupervised(token)
    features.update(features_u)

    features_ca = features_case(tweet)
    features.update(features_ca)

    features_g = features_global(token, pre_token, pre_pre_token, suc_token, suc_suc_token)
    features.update(features_g)

    features['t.ner'] = ner_tag

    return features
