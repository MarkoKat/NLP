import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
import collections
import numpy as np

# reading files
df_crew = pd.read_excel('..\\data\\Popravki - IMapBook - CREW and discussions dataset.xlsx', sheet_name='CREW data')
df_diss = pd.read_excel('..\\data\\Popravki - IMapBook - CREW and discussions dataset.xlsx',
                        sheet_name='Discussion only data')

crew_messages = df_crew['Message']
crew_class = df_crew['CodePreliminary']

message_classes = []

# ------------------------------------------

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

print(crew_dict)

crew_dict_s = {y: x for x, y in crew_dict.items()}
print(crew_dict_s)

for el in crew_class:
    el = el.lower()
    if el[-1] == " ":
        el = el[:-1]
    message_classes.append(crew_dict[el])

print(message_classes)
# ------------------------------------------

print("----")
print("First example:")
example = 1
print(crew_messages[example])
print(message_classes[example])
print(list(crew_dict.keys())[list(crew_dict.values()).index(message_classes[example])])
print(crew_dict_s[message_classes[example]])
print("----")

all_dict = {}
for code in message_classes:
    if crew_dict_s[code] not in all_dict:
        all_dict[crew_dict_s[code]] = 1
    else:
        all_dict[crew_dict_s[code]] += 1

all_dict_print = collections.OrderedDict(sorted(all_dict.items()))
print("CREW data")
print(all_dict_print)

all_dict_copy = all_dict.copy()
del_keys = []
for key in all_dict_copy:
    if all_dict_copy[key] < 10:
        del all_dict[key]
        del_keys.append(key)

print('Delete keys list:')
print(del_keys)

all_dict_print = collections.OrderedDict(sorted(all_dict.items()))
print("CREW data - cleaned")
print(all_dict_print)

print("Data length: ", len(message_classes), " - ", len(crew_messages))

crew_messages_np = np.array(crew_messages)
for i in range(len(message_classes)):
    for del_key in del_keys:
        if crew_dict_s[message_classes[i]] == del_key:
            # print(del_key)
            # print(crew_messages[i])
            message_classes[i] = None
            crew_messages_np[i] = None
            break

message_classes = list(filter(lambda a: a is not None, message_classes))
crew_messages = list(filter(lambda a: a is not None, crew_messages_np))
crew_messages_np = np.array(crew_messages)
print("Data length (after removal): ", len(message_classes), " - ", len(crew_messages_np))

# ----------------------------

# message_classes = []
#
# for el in crew_class:
#   el = el.lower()
#   if el[-1] == " ":
#     el = el[:-1]
#   message_classes.append(crew_dict[el])
#   # message_classes.append(diss_dict[el])

# message_classes = shuffle(message_classes, random_state=10)

class_train = message_classes[0:567]
class_test = message_classes[567:711]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=10)
print("StratifiedShuffleSplit - n_splits: ", sss.get_n_splits(crew_messages_np, message_classes))

# Stratified split
message_classes_np = np.array(message_classes)
for train_index, test_index in sss.split(crew_messages_np, message_classes):
    # print("TRAIN:", train_index, "TEST:", test_index)
    mess_train, mess_test = crew_messages_np[train_index], crew_messages_np[test_index]
    class_train, class_test = message_classes_np[train_index], message_classes_np[test_index]
    # print(crew_messages[train_index])
    # print(message_classes_np[test_index])

train_dict = {}
for code in class_train:
    if crew_dict_s[code] not in train_dict:
        train_dict[crew_dict_s[code]] = 1
    else:
        train_dict[crew_dict_s[code]] += 1

train_dict = collections.OrderedDict(sorted(train_dict.items()))
print("CREW data - TRAIN")
print(train_dict)

test_dict = {}
for code in class_test:
    if crew_dict_s[code] not in test_dict:
        test_dict[crew_dict_s[code]] = 1
    else:
        test_dict[crew_dict_s[code]] += 1

test_dict = collections.OrderedDict(sorted(test_dict.items()))
print("CREW data - TEST")
print(test_dict)
