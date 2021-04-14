import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# ---------------- Data preparation/pre-processing ---------------------------------------------------------------------

# reading files
df_crew = pd.read_excel('..\\data\\Popravki - IMapBook - CREW and discussions dataset.xlsx', sheet_name='CREW data')
df_diss = pd.read_excel('..\\data\\Popravki - IMapBook - CREW and discussions dataset.xlsx', sheet_name='Discussion only data')


# column CodePeliminary (CREW data)
crew_class = df_crew['CodePreliminary']

# column CodePeliminary (Discussion only data)
diss_class = df_diss['CodePreliminary']

# preparation of dictionary of types
crew_dict = {}
index = 0
for code in crew_class:
  code = code.lower()
  if code[-1] == " ":
    code = code[:-1]
  if code not in crew_dict:
    crew_dict[code] = index
    index += 1

# diss_dict = {}
# index = 0
# for code in diss_class:
#   code = code.lower()
#   if code not in diss_dict:
#     diss_dict[code] = index
#     index += 1

# for key,value in sorted(crew_dict.items()):
#   print(key, value)

print('Classes:')
print(crew_dict)
# print(diss_dict)
print('----------')

# column Message
messages = df_crew['Message']
# messages = df_diss['Message']

# column CodePeliminary
classes = df_crew['CodePreliminary']
# classes = df_diss['CodePreliminary']
message_classes = []

for el in classes:
  el = el.lower()
  if el[-1] == " ":
    el = el[:-1]
  message_classes.append(crew_dict[el])
  # message_classes.append(diss_dict[el])

# preparation of train data
mes_train = messages[0:567]
class_train = message_classes[0:567]

# preparation of test data
mes_test = messages[567:711]
class_test = message_classes[567:711]

# # preparation of train data
# mes_train = messages[0:100]
# class_train = message_classes[0:100]
#
# # preparation of test data
# mes_test = messages[100:131]
# class_test = message_classes[100:131]

# ------------------ TFIDF ----------------------------

vect = TfidfVectorizer() # parameters for tokenization, stopwords can be passed
tfidf = vect.fit_transform(mes_train)
tfidf_test = vect.transform(mes_test)

ti2 = tfidf.T.A
ti3 = list(map(list, zip(*ti2)))

X = ti3
y = class_train

# --------- Select top 'k' of the vectorized features ---------
TOP_K = 20000
selector = SelectKBest(f_classif, k=min(TOP_K, len(X[1])))
selector.fit(X, class_train)
print('----------')
try:
  x_train = selector.transform(X).astype('float32')
except RuntimeWarning as error:
  print(error)
try:
  x_test = selector.transform(tfidf_test).astype('float32')
except RuntimeWarning as error:
  print(error)

# ---------------- MLP ----------------------------------------
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)

# clf.fit(X, y)
clf.fit(x_train, y)

# ----------------- Make predictions ------------------------------------------------
print('Predictions:')
# predictions = clf.predict(tfidf_test)
predictions = clf.predict(x_test)
for p in range(len(predictions)):
  str_pred = list(crew_dict.keys())[list(crew_dict.values()).index(predictions[p])]
  str_true = list(crew_dict.keys())[list(crew_dict.values()).index(class_test[p])]
  # str_pred = list(diss_dict.keys())[list(diss_dict.values()).index(predictions[p])]
  # str_true = list(diss_dict.keys())[list(diss_dict.values()).index(class_test[p])]
  print('Predicted: ', str_pred, ' - True: ', str_true)


# ---------------------- Accuracy --------------------------------------------------
print('Accuracy:')
print(clf.score(x_test, class_test))
