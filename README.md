This is a collection of scripts used to test a Methodology for Named Entity Recognition (NER) in Short Texts such as tweets. This methodology was developed as part of my Bachelor degree Thesis. The Thesis can be found (in Spanish) in the Documentation folder.

## Thesis abstract

Abstract Short messages (like tweets and SMS) are a potentially rich source of continuously and instantly updated information. The lack of context and the informality of such messages are challenges for traditional Named Entity Recognition systems. Most eﬀorts done in this direction rely on supervised machine learning techniques which are expensive in terms of data collection and training. In this thesis we present a semi-supervised approach to Named Entity Recogniction using self-training. We use wikis as external knowledge and unsupervised features to improve portability. Whether or not the use of case in a tweet is correct is also evaluated. This avoids one of the most common problems of traditional named entity recognition systems when facing noisy environments like Twitter: an excessive dependence on title case as an indicator of the presence of an entity. The results obtained when applying this methodology ara similar to those achieved by the most popular and eﬃcient named entity recognition systems. These results validate the eﬀectiveness of the methodology.

## Features

The features extracted from each tweet can be divided into four groups.

1. Classic features

We denominate the ﬁrst group of features as "classic" features because they are the most commonly used ones in NER. This group is made up by orthographic, syntactic and grammatical features. It includes the token's sequence of characters, suffixes, prefixes, part of speech and neighbors.

2. Unsupervised features

These were extrated by joining our 7000 annotated tweets with unannotated tweets. Features are obtained as a result of three clustering algorithms: Brown clusters, Clark clusters and word2vec. These clustering algorithms tend to assign named entities that belong to the same class to the same cluster.

3. Proper case use

Due to the importance of capital letters for most entity detection systems, our corpus of annotated tweets was compared to a more formal and well formated text. This allowed us to determine for each tweet whether or not case was properly used.

4. Global fatures

It has been proven that knowledge bases and dictionaries improve NER. However, building and maintaining this knowledge bases is a costly process. We propose the use of an existing collection of knowledge: Wikipedia. 
* The ﬁrst attribute obtained using global knowledge indicates whether there exists a Wikipedia article whose title matches the token or a collocation that contains it. 
* The second of these attributes is explored only if a Wikipedia article is found. We obtain the article’s category list. We remove from this list categories containing the words: article, wikipedia and page. This categories have no semantic value. We select the category that is most common in the tweet corpus. 
* The third attribute is also only obtained if the article is found. A Wikipedia article’s ﬁrst sentence usually describes the named entity associated to it. We use as a token's description, the ﬁrst noun that follows the verb to be.
<cite>Bartolomé Maximiliano Moré (24 August 1919 – 19 February 1963), known as Benny Moré and Beny Moré (in Spanish), was a Cuban **singer**, bandleader and songwriter.</cite>

## Classifiers

Many classifiers were tested both traditional classifiers and sequence based ones. Traditional classifiers tested include: ID3, SVM, Naive Bayes, Random Forest, Stochastic Gradient Descent, and AdaBoost. Sequence based ones include: Hidden Markov Models, Maximum Entropy Models and Conditional Random Fields. Conditional Random Fields proved to be the most suitale for our problem.

## Other tests

Dimensionality reduction and data balancing algorithms were also tested.

## Corpus

The corpus used for these tests is the xLime Twitter Corpus which can be found [here](https://github.com/lrei/xlime_twitter_corpus).
*Rei, Luis, Dunja Mladenić y Simon Krek: A Multilingual Social Media Linguistic Corpus. 2014.*

## Data

The vectors resulting from the feature extraction can be found [here](https://drive.google.com/open?id=1r7ShaZNMfV9CVAostClbb8NfmjQ1ZF78).

