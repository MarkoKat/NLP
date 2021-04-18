from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


sheet_name_crew = "CREW data"
sheet_name_response = "CREW final responses"

# reading files
df_data = pd.read_excel('..\\data\\Popravki - IMapBook - CREW and discussions dataset.xlsx', sheet_name=sheet_name_crew)
df_responses = pd.read_excel('..\\data\\Popravki - IMapBook - CREW and discussions dataset.xlsx', sheet_name=sheet_name_response)

messages = df_data['Message']
classes = df_data['CodePreliminary']
mes_response_link = df_data['Response Number']
book_ids = df_data['Book ID']

# print(df_data.head())
responses = df_responses['Collab Response']
response_numbers = df_responses['Response Number']

response_dict = {}
for i in range(len(response_numbers)):
    response_dict[response_numbers[i]] = responses[i]

# print(response_dict[1.4])
# print(response_dict[mes_response_link[40]])

# TF-IDF ---------------------------------------------------------------------------------------------------------------

vect = TfidfVectorizer()  # parameters for tokenization, stopwords can be passed
tfidf_messages = vect.fit_transform(messages)
tfidf_responses = vect.transform(responses)

ti2 = tfidf_messages.T.A
tfidf_messages = list(map(list, zip(*ti2)))

ti2 = tfidf_responses.T.A
tfidf_responses = list(map(list, zip(*ti2)))

# print(tfidf_messages[0])
# print(tfidf_responses[0])
# print(vect.get_feature_names())

response_tfidf_dict = {}
for i in range(len(response_numbers)):
    response_tfidf_dict[response_numbers[i]] = tfidf_responses[i]

print("Length messages: ", len(messages), " Length tfidf_messages: ", len(tfidf_messages))
print("Length responses: ", len(responses), " Length tfidf_responses: ", len(tfidf_responses))

# print(cosine_similarity([tfidf_messages[4]], [tfidf_responses[0]]))

# Calculate average similarity with response for each class
class_dict = {}
index = 0
sim_dict = {}
count_dict = {}
for i in range(len(messages)):
    code = classes[i].lower()
    if code[-1] == " ":
        code = code[:-1]
    if code not in class_dict:
        class_dict[code] = index
        index += 1
        sim_dict[code] = 0
        count_dict[code] = 0
    sim_dict[code] = sim_dict[code] + cosine_similarity([tfidf_messages[i]], [response_tfidf_dict[mes_response_link[i]]])[0]
    # print(mes_response_link[i], " - ", response_tfidf_dict[mes_response_link[i]])
    count_dict[code] = count_dict[code] + 1

print('Classes:')
print(class_dict)

print('==============================')
print("Average similarities (responses):")
for key in sim_dict:
    sim_dict[key] = sim_dict[key] / count_dict[key]
    print(key, ": ", sim_dict[key])

# print(sim_dict)

# BOOKS ----------------------------------------------------------------------------------------------------------------
print('============================================================================')

with open('..\\data\\ID260 and ID261 - The Lady or the Tiger.txt', 'r', encoding='utf-8') as file:
    book_260_data = file.read().replace('\n', ' ')
# print(book_260_data)

with open('..\\data\\ID264 and ID265 - Just Have Less.txt', 'r', encoding='utf-8') as file:
    book_264_data = file.read().replace('\n', ' ')
# print(book_264_data)

with open('..\\data\\ID266 and ID267 - Design for the Future When the Future Is Bleak.txt', 'r', encoding='utf-8') as file:
    book_266_data = file.read().replace('\n', ' ')
# print(book_266_data)

tfidf_books = vect.transform([book_260_data, book_264_data, book_266_data])
ti2 = tfidf_books.T.A
tfidf_books = list(map(list, zip(*ti2)))

# print(cosine_similarity([tfidf_messages[1]], [tfidf_books[1]]))
# print(tfidf_books)

book_dict = {
    260: 0,
    261: 0,
    264: 1,
    265: 1,
    266: 2,
    267: 2
}

# print(book_ids)
# print(tfidf_books[book_dict[book_ids[0]]])

# Calculate average similarity with response for each class
class_dict = {}
index = 0
sim_dict = {}
count_dict = {}
for i in range(len(messages)):
    if type(book_ids[i]) == int:
        code = classes[i].lower()
        if code[-1] == " ":
            code = code[:-1]
        if code not in class_dict:
            class_dict[code] = index
            index += 1
            sim_dict[code] = 0
            count_dict[code] = 0
        sim_dict[code] = sim_dict[code] + cosine_similarity([tfidf_messages[i]], [tfidf_books[book_dict[book_ids[i]]]])[0]
        # print(mes_response_link[i], " - ", response_tfidf_dict[mes_response_link[i]])
        count_dict[code] = count_dict[code] + 1

print("Average similarities (books):")
for key in sim_dict:
    sim_dict[key] = sim_dict[key] / count_dict[key]
    print(key, ": ", sim_dict[key])
