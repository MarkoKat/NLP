import nltk
import string
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


def get_tokens(text):
    text = text.lower()
    # remove punctuation
    table = text.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    tokens = nltk.word_tokenize(text)
    return tokens


def get_stemms(texts):
    stemmer = SnowballStemmer("english")
    tokens_stem = []

    for text in texts:
        stems = [stemmer.stem(token) for token in get_tokens(text)]
        tokens_stem.append(stems)

    sentences_stem = []
    for sentence in tokens_stem:
        sentences_stem.append(' '.join(sentence))

    return sentences_stem


def get_lemmas(texts):
    lemmatizer = WordNetLemmatizer()
    tokens_lemma = []

    for text in texts:
        lemmas = [lemmatizer.lemmatize(token) for token in get_tokens(text)]
        tokens_lemma.append(lemmas)

    sentences_lemma = []
    for sentence in tokens_lemma:
        sentences_lemma.append(' '.join(sentence))

    return sentences_lemma


if __name__ == "__main__":
    texts = ["My car is really fast",
             "I like driving cars",
             "This flower are beautiful",
             "Taking care of flowers is realxing",
             "Cars are my hobby",
             "Flowers are my hobby"]

    sentences_stem = get_stemms(texts)
    print(sentences_stem)

    sentences_lemma = get_lemmas(texts)
    print(sentences_lemma)

    # TF-IDF -----------------------------------------------------------------------------------------------------------

    print('TF-IDF feature names:')

    vect_b = TfidfVectorizer()  # parameters for tokenization, stopwords can be passed
    tfidf_messages_b = vect_b.fit_transform(texts)
    print(vect_b.get_feature_names())

    # Stemming
    vect = TfidfVectorizer()  # parameters for tokenization, stopwords can be passed
    tfidf_messages = vect.fit_transform(sentences_stem)
    print(vect.get_feature_names())

    # Lemmatisation
    vect_l = TfidfVectorizer()  # parameters for tokenization, stopwords can be passed
    tfidf_messages_l = vect_l.fit_transform(sentences_lemma)
    print(vect_l.get_feature_names())

