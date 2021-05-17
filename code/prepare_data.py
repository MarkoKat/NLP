import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


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


def get_data(sheet, use_response_similarity, use_book_similarity):
    # ---------------- Data preparation/pre-processing -----------------------------------------------------------------

    file_name = '../data/Popravki - IMapBook - CREW and discussions dataset.xlsx'

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
        # print("TRAIN:", train_index, "TEST:", test_index)
        mes_train, mes_test = messages[train_index], messages[test_index]
        class_train, class_test = message_classes_np[train_index], message_classes_np[test_index]
        if use_book_similarity:
            book_idx_train, book_idx_test = book_ids[train_index], book_ids[test_index]
        if use_response_similarity:
            response_link_train, response_link_test = response_link[train_index], response_link[test_index]

    return mes_train, mes_test, class_train, class_test, book_idx_train, book_idx_test, response_link_train, response_link_test, crew_dict_s
