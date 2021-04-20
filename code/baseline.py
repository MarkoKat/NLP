import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
import collections
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from stemming import get_stemms, get_lemmas
from stop_words import remove_stopwords
from similarity_analyse import get_response_tfidf_dict, get_tfidf_books, get_book_dict
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Data preparation/pre-processing ---------------------------------------------------------------------

sheet_name_crew = "CREW data"
sheet_name_diss = "Discussion only data"

# reading files
df_data = pd.read_excel('..\\data\\Popravki - IMapBook - CREW and discussions dataset.xlsx', sheet_name=sheet_name_crew)


# column Message
messages = df_data['Message']

# column CodePeliminary
classes = df_data['CodePreliminary']

# Links to responses associated with each message
response_link = df_data['Response Number']

# Book id for each message
book_ids = df_data['Book ID']

# preparation of dictionary of types
class_dict = {}
index = 0
for code in classes:
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
print(crew_dict_s)

print('----------')

# messages, classes = shuffle(messages, classes, random_state=10)

message_classes = []

for el in classes:
  el = el.lower()
  if el[-1] == " ":
    el = el[:-1]
  message_classes.append(class_dict[el])

print(message_classes)

# Remove classes with small number of samples

all_dict = {}
for code in message_classes:
    if crew_dict_s[code] not in all_dict:
        all_dict[crew_dict_s[code]] = 1
    else:
        all_dict[crew_dict_s[code]] += 1

all_dict_print = collections.OrderedDict(sorted(all_dict.items()))
print("Class counts")
print(all_dict_print)

all_dict_copy = all_dict.copy()
del_keys = []
for key in all_dict_copy:
    if all_dict_copy[key] < 5:
        del all_dict[key]
        del_keys.append(key)

print('Delete keys list:')
print(del_keys)

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

# Stemming/lemmatisation

print('MESSAGES ------------------------------------------------------')
print(messages[:10])

# print('STEMMED MESSAGES ----')
# messages = get_lemmas(messages)
# messages = np.array(messages)
# print(messages[:10])

# print('STOPWORDS REMOVED:')
# messages = remove_stopwords(messages)
# messages = np.array(messages)
# print(messages[:10])

# -------------------------------------------

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
print("StratifiedShuffleSplit - n_splits: ", sss.get_n_splits(messages, message_classes))

# Stratified split
message_classes_np = np.array(message_classes)
for train_index, test_index in sss.split(messages, message_classes):
    # print("TRAIN:", train_index, "TEST:", test_index)
    mes_train, mes_test = messages[train_index], messages[test_index]
    class_train, class_test = message_classes_np[train_index], message_classes_np[test_index]
    book_idx_train, book_idx_test = book_ids[train_index], book_ids[test_index]
    response_link_train, response_link_test = response_link[train_index], response_link[test_index]
    # print(crew_messages[train_index])
    # print(message_classes_np[test_index])

print("Train length: ", len(class_train), " Test length: ", len(class_test))

# ------------------ TFIDF ----------------------------

vect = TfidfVectorizer(min_df=2)  # parameters for tokenization, stopwords can be passed
tfidf = vect.fit_transform(mes_train)
tfidf_test = vect.transform(mes_test)

ti2 = tfidf.T.A
messages_tfidf = list(map(list, zip(*ti2)))

ti3 = tfidf_test.T.A
tfidf_test = list(map(list, zip(*ti3)))

print("TF-IDF feature names: ", vect.get_feature_names())
print("Number of TF-IDF features: ", len(vect.get_feature_names()))

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

clf.fit(x_train, class_train)
print("Number of iterations: ", clf.n_iter_)

# ----------------- Make predictions ------------------------------------------------
print('Predictions:')
# predictions = clf.predict(tfidf_test)
predictions = clf.predict(x_test)
for p in range(len(predictions)):
  str_pred = crew_dict_s[predictions[p]]
  str_true = crew_dict_s[class_test[p]]
  # print('Predicted: ', str_pred, ' - True: ', str_true)

# ---------------------- Accuracy --------------------------------------------------
print('Accuracy:')
print(clf.score(x_test, class_test))

print('Accuracy2:')
print(metrics.accuracy_score(class_test, predictions))

print('F1 (micro):')
print(metrics.f1_score(class_test, predictions, average='micro'))

print('F1 (macro):')
print(metrics.f1_score(class_test, predictions, average='macro'))

# print('Confusion matrix --------------------------------')
# print(metrics.confusion_matrix(class_test, predictions))

print('Classification report ---------------------------')
print(metrics.classification_report(class_test, predictions, digits=3))

# Probabilities ----------------------------------------------------------
example_index = 3
# print("True: ", class_test[0:10])
# pred_tst = clf.predict(x_test[0:10])
# print(pred_tst)
# prob_tst = clf.predict_proba(x_test[0:10])
# print(prob_tst)

pred_train = clf.predict(x_train)
pred_test = clf.predict(x_test)
prob_train = clf.predict_proba(x_train)
prob_test = clf.predict_proba(x_test)
# print(prob_train[0:5])
# print(pred_train[0:5])
# print(class_train[0:5])

# Similarities -----------------------------------------------------------
print("--- SIMILARITIES ---")

response_tfidf_dict = get_response_tfidf_dict(vect)

tfidf_books = get_tfidf_books(vect)
book_dict = get_book_dict()

book_idx_train = np.array(book_idx_train)
book_idx_test = np.array(book_idx_test)

response_link_train = np.array(response_link_train)
response_link_test = np.array(response_link_test)

print(book_idx_test[0:10])
ex_similarity = cosine_similarity([x_test[example_index]], [tfidf_books[book_dict[book_idx_test[example_index]]]])
print("Example similarity: ", ex_similarity)

rf_arr_train = []
for i in range(len(pred_train)):
    similarity_response = cosine_similarity([x_train[i]], [response_tfidf_dict[response_link_train[i]]])[0][0]

    similarity_book = cosine_similarity([x_train[i]], [tfidf_books[book_dict[book_idx_train[i]]]])[0][0]
    # rf_arr_train.append([pred_train[i], similarity_book])

    # prob_list = prob_train[i].tolist()
    # prob_list.append(similarity_response)
    # prob_list.append(similarity_book)
    # rf_arr_train.append(prob_list)

    # rf_arr_train.append([pred_train[i], similarity_response])
    rf_arr_train.append([pred_train[i], similarity_response, similarity_book])
print(rf_arr_train)

rf_arr_test = []
for i in range(len(pred_test)):
    similarity_response = cosine_similarity([x_test[i]], [response_tfidf_dict[response_link_test[i]]])[0][0]

    similarity_book = cosine_similarity([x_test[i]], [tfidf_books[book_dict[book_idx_test[i]]]])[0][0]
    # rf_arr_test.append([pred_test[i], similarity_book])

    # prob_list = prob_test[i].tolist()
    # prob_list.append(similarity_response)
    # prob_list.append(similarity_book)
    # rf_arr_test.append(prob_list)

    # rf_arr_test.append([pred_test[i], similarity_response])
    rf_arr_test.append([pred_test[i], similarity_response, similarity_book])
print(rf_arr_test)

clf_rf = RandomForestClassifier(max_depth=10, random_state=0)
clf_rf.fit(rf_arr_train, class_train)

# print("True: ", class_test[example_index])
# print("MLP prediction: ", predictions[example_index])
# rf_pred_ex = clf_rf.predict([[predictions[example_index], 0]])
# print("RF prediction:", rf_pred_ex)

rf_pred = clf_rf.predict(rf_arr_test)

print('Classification report - Random forest ---------------------------')
print(metrics.classification_report(class_test, rf_pred, digits=3))
