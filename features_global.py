from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag.stanford import StanfordPOSTagger

import wikipedia
import string

search_history = {}
category_history = {}
spanish_stopwords = stopwords.words('spanish')
english_stopwords = stopwords.words('english')

tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger', 'stanford-postagger.jar')


def features_global(token, pre_token, pre_pre_token, suc_token, suc_suc_token):
    features = {}
    features['has_wiki_page'] = False

    punct = list(string.punctuation) + ['¿', '¡', '...', '..']

    if token in spanish_stopwords or token.isdigit() or token in punct:
        return features

    # page = most_relevant_page(token, [pre_token, pre_pre_token, suc_token, suc_suc_token])
    token = unidecode(token.lower())

    if not token:
        return features

    result = [r for r in wikipedia.search(token) if token in unidecode(r.lower())]

    if not result:
        return features

    if result:
        page = result[0]
        features['has_wiki_page'] = True
        features['wiki_page'] = page

        try:
            wiki_page = wikipedia.page(page)

            description = first_sentence_description(token, page)

            if description:
                features['wiki_description'] = description.lower()

            categories = wiki_page.categories

            categories = [c for c in categories if
                          'wiki' not in c.lower() and 'article' not in c.lower()
                          and 'page' not in c.lower() and not c.isdigit() and c != '' and c != ' '
                          and len(c) != 1 and not 'language' in c.lower() and not 'cs1' in c.lower()
                          and not 'words' in c.lower() and not 'links' in c.lower() and not 'grammar' in c.lower()]

            if categories:
                features['wiki_category'] = categories

            return features

        except wikipedia.WikipediaException:
            return features

    return features


def most_relevant_page(token, window):
    bigrams = open('collocations/bigrams')
    max_pmi = float(bigrams.readline().split()[-1])
    bigrams = open('collocations/bigrams')

    relevant_articles = []
    line = token.split()

    while line:
        if token in line:
            score = float(line[-1]) / max_pmi if len(line) > 1 else 1
            score += sum([w in line for w in window])

            query = ' '.join(line[:-1] if len(line) > 1 else line)
            result = search_history[query] if query in search_history else wikipedia.search(query)
            articles = [article for article in result if unidecode(token.lower()) in unidecode(article[0].lower())]

            for article in articles:
                unicode_article = unidecode(article[0].lower())
                article_score = score + sum([unidecode(w.lower()) in unicode_article for w in window if w]) + int(
                    token in article)
                relevant_articles.append((article, article_score))

        line = bigrams.readline().split()

    if not relevant_articles:
        return None

    max_score = max(relevant_articles, key=lambda x: x[1])[1]
    max_articles = [p for p in relevant_articles if p[1] == max_score]
    max_with_token = [p for p in max_articles if token in p[0]]

    bigrams.close()

    return min(max_with_token if max_with_token else max_articles, key=lambda x: len(x[0]))


def first_sentence_description(token, page):
    first_sentence = wikipedia.summary(page, sentences=1)
    first_sentence = word_tokenize(first_sentence, language='english')

    first_sentence_tagged = tagger.tag(first_sentence)

    to_be = open('vocabulary/to_be_english').read().split('\n')

    to_be_in_sentence = [word for word in first_sentence if word in to_be]

    if to_be_in_sentence:
        words = first_sentence_tagged[first_sentence.index(to_be_in_sentence[0]) + 1:]
        words = [w for w in words if w[0].lower() not in english_stopwords]
        nouns = [w for w in words if w[1]=='NN' or w[1]=='NNS']
        if nouns:
            return nouns[0][0]
        return words[0][0] if words else None
    return None


def most_relevant_category(token, page):
    categories = wikipedia.page(page).categories
    frequency = {k: v for k, v in category_history.items() if k in categories}
    return max(frequency, key=frequency.get)

# features_global('chavez', 'hugo', 'presidente', 'viva', '')
