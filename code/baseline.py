import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


df_crew = pd.read_excel('..\\data\\IMapBook - CREW and discussions dataset.xlsx', sheet_name='CREW data')
df_diss = pd.read_excel('..\\data\\IMapBook - CREW and discussions dataset.xlsx', sheet_name='Discussion only data')

diss_class = df_diss['CodePreliminary']

diss_dict = {}

index = 0
for code in diss_class:
  code = code.lower()
  if code not in diss_dict:
    diss_dict[code] = index
    index += 1

print('Classes:')
print(diss_dict)
print('----------')

messages = df_diss['Message']

classes = df_diss['CodePreliminary']
message_classes = []

for el in classes:
  el = el.lower()
  message_classes.append(diss_dict[el])

mes_train = messages[0:100]
class_train = message_classes[0:100]

mes_test = messages[100:131]
class_test = message_classes[100:131]

# ----------------------------------------------

vect = TfidfVectorizer() # parameters for tokenization, stopwords can be passed
tfidf = vect.fit_transform(mes_train)
tfidf_test = vect.transform(mes_test)

ti2 = tfidf.T.A
ti3 = list(map(list, zip(*ti2)))

X = ti3
y = class_train

# Select top 'k' of the vectorized features. ---------
TOP_K = 20000
selector = SelectKBest(f_classif, k=min(TOP_K, len(X[1])))
selector.fit(X, class_train)
x_train = selector.transform(X).astype('float32')
x_test = selector.transform(tfidf_test).astype('float32')

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)

# clf.fit(X, y)
clf.fit(x_train, y)

# Make predictions
print('Predictions:')
# predictions = clf.predict(tfidf_test)
predictions = clf.predict(x_test)
for p in range(len(predictions)):
  str_pred = list(diss_dict.keys())[list(diss_dict.values()).index(predictions[p])]
  str_true = list(diss_dict.keys())[list(diss_dict.values()).index(class_test[p])]
  print('Predicted: ', str_pred, ' - True: ', str_true)

print('Accuracy:')
print(clf.score(x_test, class_test))
