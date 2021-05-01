import numpy as np
from sklearn.utils import shuffle
import pandas as pd

# reading files
df_crew = pd.read_excel('..\\..\\data\\Popravki - IMapBook - CREW and discussions dataset.xlsx', sheet_name='CREW data')
df_diss = pd.read_excel('..\\..\\data\\Popravki - IMapBook - CREW and discussions dataset.xlsx', sheet_name='Discussion only data')

crew_class = df_crew['CodePreliminary']
crew_message = df_crew['Message']

# X = [[1., 0.], [2., 1.], [0., 0.]]
# y = [0, 1, 2]

crew_message_sh, crew_class_sh = shuffle(crew_message, crew_class, random_state=10)

print(crew_class_sh[0:10])
print(crew_message_sh[0:10])

print("-----------")

print(crew_class_sh[700:711])
print(crew_message_sh[700:711])
