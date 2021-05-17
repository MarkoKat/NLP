from nltk import word_tokenize
import nltk
import re
import string
import numpy as np
import pandas as pd
from emoji import UNICODE_EMOJI
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import words
from sentence_transformers import SentenceTransformer

from prepare_data import get_data
from similarities import use_similarities
from tfidf import get_tfidf_vectors
from confusion_matrix import get_confusion_matrix


def num_words(tokens):
    return len(tokens)


def frequency5w1h(tokens):
    wh_count = 0
    for token in tokens:
        token = token.lower()
        if token.startswith("w") or token.startswith("h"):
            wh_count += 1
    return wh_count / len(tokens)


def frequency_upper(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    upper_count = 0
    for letter in text:
        if letter.isupper():
            upper_count += 1
    if len(text) == 0:
        return 0
    return upper_count / len(text)


def has_link(text):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, text)
    if len(url) > 0:
        return 1
    return 0


def is_ok(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower().strip()
    if text == "ok":
        return 1
    return 0


def is_emoji(text):
    text = text.strip()
    if text == ":)" or text == ":(" or text == ":D":
        return 1
    if text in UNICODE_EMOJI['en']:
        return 1
    return 0


def has_emoji(text, tokens):
    if ":)" in text or ":(" in text or ":D" in text:
        return 1
    for token in tokens:
        if is_emoji(token):
            return 1
    return 0


def has_question_mark(text):
    if "?" in text:
        return 1
    return 0


def has_exclamation_mark(text):
    if "!" in text:
        return 1
    return 0


def frequency_mistakes(tokens):
    mis_count = 0
    words_count = 0
    for token in tokens:
        if token not in string.punctuation:
            words_count += 1
            if token.lower() not in words.words():
                mis_count += 1
    if words_count == 0:
        return 0
    return mis_count / words_count


def max_word_length(tokens):
    max_len = 0
    for token in tokens:
        if len(token) > max_len:
            max_len = len(token)
    return max_len


def average_word_length(tokens):
    if len(tokens) == 0:
        return 0
    sum_length = 0
    for token in tokens:
        sum_length += len(token)
    return sum_length / len(tokens)


def has_greeting(text):
    if text.lower().startswith("hi") or text.lower().startswith("hello") or text.lower().startswith("good morning"):
        return 1
    return 0


def has_thank(text):
    if "thank" in text:
        return 1
    return 0


def get_feature_vect(message_array):
    message_feature_vect = []
    for message in message_array:
        mes_tokens = word_tokenize(message)

        feature_vect = []
        feature_vect.append(num_words(mes_tokens))
        feature_vect.append(frequency5w1h(mes_tokens))
        feature_vect.append(frequency_upper(message))
        feature_vect.append(has_link(message))
        feature_vect.append(is_ok(message))
        feature_vect.append(is_emoji(message))
        feature_vect.append(has_emoji(message, mes_tokens))
        feature_vect.append(has_question_mark(message))
        feature_vect.append(has_exclamation_mark(message))
        # feature_vect.append(frequency_mistakes(mes_tokens))
        feature_vect.append(max_word_length(mes_tokens))
        feature_vect.append(average_word_length(mes_tokens))
        feature_vect.append(has_greeting(message))
        feature_vect.append(has_thank(message))

        message_feature_vect.append(feature_vect)
    return message_feature_vect


if __name__ == "__main__":
    print("Test manual features")
    nltk.download('punkt')

    test_text = "Who is in MY ROOM rroom?"
    # test_text = "https://usflearn.instructure.com/courses/1454200/pages/part-2-study-of-online-discussions-deeper-w-slash-less-work-please-open?module_item_id=19552705"
    # test_text = "OK, "
    # test_text = "heh 👍"
    # test_text = "Erica :D"
    # test_text = "Erica <3"

    print(test_text)
    test_tokens = word_tokenize(test_text)
    print(test_tokens)

    # print("num_words: ", num_words(test_tokens))
    # print("frequency5w1h: ", frequency5w1h(test_tokens))
    # print("frequency_upper: ", frequency_upper(test_text))
    # print("has_link: ", has_link(test_text))
    # print("is_ok: ", is_ok(test_text))
    # print("is_emoji: ", is_emoji(test_text))
    # print("has_emoji: ", has_emoji(test_text, test_tokens))
    # print("has_question_mark: ", has_question_mark(test_text))
    # print("frequency_mistakes: ", frequency_mistakes(test_tokens))
    # print("max_word_length: ", max_word_length(test_tokens))
    # print("average_word_length: ", average_word_length(test_tokens))
    #
    # print("---------------------")

    # sheet = 'crew'
    sheet = 'discussion'

    use_response_similarity = False  # Can't use with discussion
    use_book_similarity = False

    use_bert = False

    # Read command line arguments
    arguments = sys.argv
    print("Arguments: ", arguments)

    if 'crew' in arguments:
        sheet = 'crew'
    if 'discussion' in arguments:
        sheet = 'discussion'
    if 'use_response_similarity' in arguments:
        use_response_similarity = True
    if 'use_book_similarity' in arguments:
        use_book_similarity = True
    if 'use_bert_for_similarity' in arguments:
        use_bert = True

    # Get data
    mes_train, mes_test, class_train, class_test, book_idx_train, book_idx_test, response_link_train, response_link_test, class_dict = get_data(
        sheet, use_response_similarity, use_book_similarity)

    print("Train length: ", len(class_train), " Test length: ", len(class_test))

    # Prepare feature vectors
    train_vect = get_feature_vect(mes_train)
    test_vect = get_feature_vect(mes_test)

    clf = MLPClassifier(solver='adam', alpha=1e-5, activation='relu', max_iter=5000,
                        hidden_layer_sizes=(20), random_state=1, learning_rate='constant')

    # Train MLP
    clf.fit(train_vect, class_train)
    print("Number of iterations: ", clf.n_iter_)

    predictions = clf.predict(test_vect)

    print('Classification report ---------------------------')
    print(metrics.classification_report(class_test, predictions, digits=3))

    # Confusion matrix
    class_names = [class_dict[x] for x in list(set(class_test))]
    get_confusion_matrix(class_test, predictions, class_names)

    # --- Similarities -----------------------------------------------------------
    x_train, x_test, tfidf_vectorizer = get_tfidf_vectors(mes_train, mes_test)

    if use_response_similarity or use_book_similarity:
        print("--- SIMILARITIES ---")

        pred_train = class_train
        pred_test = predictions

        if use_bert:
            bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
            x_train = bert_model.encode(mes_train)
            x_test = bert_model.encode(mes_test)

        use_similarities(use_response_similarity, use_book_similarity, tfidf_vectorizer, x_train, x_test,
                         pred_train, pred_test, class_train, class_test,
                         book_idx_train, book_idx_test, response_link_train, response_link_test, use_bert, class_names)
