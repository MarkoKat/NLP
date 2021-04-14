import pandas as pd
from sklearn.utils import shuffle


# reading files
df_crew = pd.read_excel('..\\data\\Popravki - IMapBook - CREW and discussions dataset.xlsx', sheet_name='CREW data')
df_diss = pd.read_excel('..\\data\\Popravki - IMapBook - CREW and discussions dataset.xlsx', sheet_name='Discussion only data')

crew_class = df_crew['CodePreliminary']

crew_dict = {}

for code in crew_class:
    code = code.lower()
    if code.endswith(" "):
        code = code[:-1]

    if code not in crew_dict:
        crew_dict[code] = 1
    else:
        crew_dict[code] += 1

print("CREW data")
print(crew_dict)

# ----------------------------

message_classes = []

for el in crew_class:
  el = el.lower()
  if el[-1] == " ":
    el = el[:-1]
  message_classes.append(crew_dict[el])
  # message_classes.append(diss_dict[el])

crew_class = shuffle(crew_class, random_state=10)

class_train = crew_class[0:567]
class_test = crew_class[567:711]

crew_dict = {}

for code in class_train:
    code = code.lower()
    if code.endswith(" "):
        code = code[:-1]

    if code not in crew_dict:
        crew_dict[code] = 1
    else:
        crew_dict[code] += 1

print("CREW data - TRAIN")
print(crew_dict)

crew_dict = {}
for code in class_test:
    code = code.lower()
    if code.endswith(" "):
        code = code[:-1]

    if code not in crew_dict:
        crew_dict[code] = 1
    else:
        crew_dict[code] += 1

print("CREW data - TEST")
print(crew_dict)
