from nltk import word_tokenize
import re
import string
import numpy as np
import pandas as pd
from emoji import UNICODE_EMOJI
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

from baseline import get_class_dict, get_message_classes, remove_small_classes


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


if __name__ == "__main__":
    print("Test manual features")

    test_text = "Who is in MY ROOM?"
    # test_text = "https://usflearn.instructure.com/courses/1454200/pages/part-2-study-of-online-discussions-deeper-w-slash-less-work-please-open?module_item_id=19552705"
    # test_text = "OK, "
    # test_text = "heh 👍"
    # test_text = "Erica :D"
    # test_text = "Erica <3"

    print(test_text)
    test_tokens = word_tokenize(test_text)
    print(test_tokens)

    print("num_words: ", num_words(test_tokens))
    print("frequency5w1h: ", frequency5w1h(test_tokens))
    print("frequency_upper: ", frequency_upper(test_text))
    print("has_link: ", has_link(test_text))
    print("is_ok: ", is_ok(test_text))
    print("is_emoji: ", is_emoji(test_text))
    print("has_emoji: ", has_emoji(test_text, test_tokens))
    print("has_question_mark: ", has_question_mark(test_text))

    print("---------------------")

    # sheet = 'crew'
    sheet = 'discussion'

    use_response_similarity = False  # Can't use with discussion
    use_book_similarity = False

    # ---------------- Data preparation/pre-processing -----------------------------------------------------------------

    file_name = '..\\data\\Popravki - IMapBook - CREW and discussions dataset.xlsx'

    if sheet == 'crew':
        sheet_name = "CREW data"
    else:
        sheet_name = "Discussion only data"

    # reading files
    df_data = pd.read_excel(file_name, sheet_name=sheet_name)

    # column Message
    messages = df_data['Message']

    # column CodePreliminary
    classes = df_data['CodePreliminary']

    # Links to responses associated with each message
    response_link = None
    if use_response_similarity:
        response_link = df_data['Response Number']

    # Book id for each message
    book_ids = None
    if use_book_similarity:
        book_ids = df_data['Book ID']

    # ----------------------------------------------------------------------
    class_dict, crew_dict_s = get_class_dict(classes)
    print('----------')

    message_classes = get_message_classes(classes, class_dict)

    # Remove classes with small number of samples
    messages, message_classes = remove_small_classes(messages, message_classes, 5, crew_dict_s)

    # --- Split data to train-test -------------------------------------------

    # preparation of train data
    mes_train = None
    class_train = None
    book_idx_train = None
    response_link_train = None

    # preparation of test data
    mes_test = None
    class_test = None
    book_idx_test = None
    response_link_test = None

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=10)
    # print("StratifiedShuffleSplit - n_splits: ", sss.get_n_splits(messages, message_classes))

    # Stratified split
    message_classes_np = np.array(message_classes)
    for train_index, test_index in sss.split(messages, message_classes):
        # print("TRAIN:", train_index, "TEST:", test_index)
        mes_train, mes_test = messages[train_index], messages[test_index]
        class_train, class_test = message_classes_np[train_index], message_classes_np[test_index]
        if use_book_similarity:
            book_idx_train, book_idx_test = book_ids[train_index], book_ids[test_index]
        if use_response_similarity:
            response_link_train, response_link_test = response_link[train_index], response_link[test_index]

    print("Train length: ", len(class_train), " Test length: ", len(class_test))

    # Prepare feature vectors
    train_vect = []
    test_vect = []

    for message in mes_train:
        mes_tokens = word_tokenize(message)

        feature_vect = []
        feature_vect.append(num_words(mes_tokens))
        feature_vect.append(frequency5w1h(mes_tokens))
        feature_vect.append(frequency_upper(message))
        feature_vect.append(has_link(message))
        feature_vect.append(is_ok(message))
        # feature_vect.append(is_emoji(message))
        # feature_vect.append(has_emoji(message, mes_tokens))
        feature_vect.append(has_question_mark(message))

        train_vect.append(feature_vect)

    # print(train_vect)

    for message in mes_test:
        mes_tokens = word_tokenize(message)

        feature_vect = []
        feature_vect.append(num_words(mes_tokens))
        feature_vect.append(frequency5w1h(mes_tokens))
        feature_vect.append(frequency_upper(message))
        feature_vect.append(has_link(message))
        feature_vect.append(is_ok(message))
        # feature_vect.append(is_emoji(message))
        # feature_vect.append(has_emoji(message, mes_tokens))
        feature_vect.append(has_question_mark(message))

        test_vect.append(feature_vect)

    clf = MLPClassifier(solver='adam', alpha=1e-5, activation='relu', max_iter=5000,
                        hidden_layer_sizes=(20), random_state=1, learning_rate='constant')

    # Train MLP
    clf.fit(train_vect, class_train)
    print("Number of iterations: ", clf.n_iter_)

    predictions = clf.predict(test_vect)

    print('Classification report ---------------------------')
    print(metrics.classification_report(class_test, predictions, digits=3))
