import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from stemming import get_stemms, get_lemmas
from stop_words import remove_stopwords
from similarity_analyse_crew import get_response_tfidf_dict, get_tfidf_books, get_book_dict
from similarities import use_similarities


def get_class_dict(classes_f):
    """Get dictionary for string and numerical form of classes"""
    class_dict = {}
    index = 0
    for code in classes_f:
        code = code.lower()
        if code[-1] == " ":
            code = code[:-1]
        if code not in class_dict:
            class_dict[code] = index
            index += 1

    print('Classes:')
    print(class_dict)

    # Switch keys and values in dict
    crew_dict_s = {y: x for x, y in class_dict.items()}
    # print(crew_dict_s)

    return class_dict, crew_dict_s


def get_message_classes(classes_f, class_dict):
    """Get array with class for each message in numerical form"""
    message_classes = []

    for el in classes_f:
        el = el.lower()
        if el[-1] == " ":
            el = el[:-1]
        message_classes.append(class_dict[el])
    return message_classes


def remove_small_classes(messages, message_classes, min_number_of_messages, crew_dict_s):
    """Remove messages from classes with number of instances smaller than min_number_of_messages"""

    all_dict = {}
    for code in message_classes:
        if crew_dict_s[code] not in all_dict:
            all_dict[crew_dict_s[code]] = 1
        else:
            all_dict[crew_dict_s[code]] += 1

    # all_dict_print = collections.OrderedDict(sorted(all_dict.items()))
    # print("Class counts")
    # print(all_dict_print)

    all_dict_copy = all_dict.copy()
    del_keys = []
    for key in all_dict_copy:
        if all_dict_copy[key] < min_number_of_messages:
            del all_dict[key]
            del_keys.append(key)

    print('Delete keys list: ', del_keys)

    print("Data length: ", len(message_classes), " - ", len(messages))

    messages_np = np.array(messages)
    for i in range(len(message_classes)):
        for del_key in del_keys:
            if crew_dict_s[message_classes[i]] == del_key:
                # print(del_key)
                # print(crew_messages[i])
                message_classes[i] = None
                messages_np[i] = None
                break

    message_classes = list(filter(lambda a: a is not None, message_classes))
    messages = list(filter(lambda a: a is not None, messages_np))
    messages = np.array(messages)
    print("Data length (after removal): ", len(message_classes), " - ", len(messages))
    return messages, message_classes


def get_predictions(clf_f, train_data, test_data):
    pred_train = clf_f.predict(train_data)
    pred_test = clf_f.predict(test_data)
    return pred_train, pred_test


def get_probabilities(clf_f, train_data, test_data):
    prob_train = clf_f.predict_proba(train_data)
    prob_test = clf_f.predict_proba(test_data)
    return prob_train, prob_test


# ======================================================================================================================
if __name__ == "__main__":

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

    # --- Stemming/lemmatisation/stop word removal ------------------------------

    # print('STEMMED MESSAGES ----')
    # messages = get_stemms(messages)
    # messages = np.array(messages)
    # print(messages[:10])

    # print('LEMMATISED MESSAGES ----')
    # messages = get_lemmas(messages)
    # messages = np.array(messages)
    # print(messages[:10])

    # print('STOPWORDS REMOVED:')
    # messages = remove_stopwords(messages)
    # messages = np.array(messages)
    # print(messages[:10])

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
        print("TRAIN:", train_index, "TEST:", test_index)
        mes_train, mes_test = messages[train_index], messages[test_index]
        class_train, class_test = message_classes_np[train_index], message_classes_np[test_index]
        if use_book_similarity:
            book_idx_train, book_idx_test = book_ids[train_index], book_ids[test_index]
        if use_response_similarity:
            response_link_train, response_link_test = response_link[train_index], response_link[test_index]

    print("Train length: ", len(class_train), " Test length: ", len(class_test))

    # ------------------ TFIDF ----------------------------

    tfidf_vectorizer = TfidfVectorizer(min_df=2)  # parameters for tokenization, stopwords can be passed
    tfidf = tfidf_vectorizer.fit_transform(mes_train)
    tfidf_test = tfidf_vectorizer.transform(mes_test)

    ti2 = tfidf.T.A
    messages_tfidf = list(map(list, zip(*ti2)))

    ti3 = tfidf_test.T.A
    tfidf_test = list(map(list, zip(*ti3)))

    print("TF-IDF feature names: ", tfidf_vectorizer.get_feature_names())
    print("Number of TF-IDF features: ", len(tfidf_vectorizer.get_feature_names()))

    # --------- Select top 'k' of the vectorized features ---------

    # TOP_K = 20000
    # selector = SelectKBest(f_classif, k=min(TOP_K, len(messages_tfidf[1])))
    # selector.fit(messages_tfidf, class_train)
    # print('----------')
    # try:
    #   x_train = selector.transform(messages_tfidf).astype('float32')
    # except RuntimeWarning as error:
    #   print(error)
    # try:
    #   x_test = selector.transform(tfidf_test).astype('float32')
    # except RuntimeWarning as error:
    #   print(error)

    x_train = messages_tfidf
    x_test = tfidf_test

    # ---------------- MLP ----------------------------------------
    clf = MLPClassifier(solver='adam', alpha=1e-5, activation='relu', max_iter=5000,
                        hidden_layer_sizes=(20), random_state=1, learning_rate='constant')

    # clf = RandomForestClassifier(max_depth=5, random_state=0)

    # Train MLP
    clf.fit(x_train, class_train)
    print("Number of iterations: ", clf.n_iter_)

    # --- Make predictions ------------------------------------------------
    predictions = clf.predict(x_test)

    # print('Predictions:')
    # for p in range(len(predictions)):
    #     str_pred = crew_dict_s[predictions[p]]
    #     str_true = crew_dict_s[class_test[p]]
    #     print('Predicted: ', str_pred, ' - True: ', str_true)

    # --- Evaluation --------------------------------------------------

    # print('Confusion matrix --------------------------------')
    # print(metrics.confusion_matrix(class_test, predictions))

    print('Classification report ---------------------------')
    print(metrics.classification_report(class_test, predictions, digits=3))

    # =============================================================================================================

    # --- Similarities -----------------------------------------------------------
    if use_response_similarity or use_book_similarity:
        print("--- SIMILARITIES ---")

        pred_train, pred_test = get_predictions(clf, x_train, x_test)
        prob_train, prob_test = get_probabilities(clf, x_train, x_test)

        use_similarities(use_response_similarity, use_book_similarity, tfidf_vectorizer, x_train, x_test,
                         pred_train, pred_test, class_train, class_test,
                         book_idx_train, book_idx_test, response_link_train, response_link_test)
