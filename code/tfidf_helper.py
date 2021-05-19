from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectors(mes_train, mes_test):
    tfidf_vectorizer = TfidfVectorizer(min_df=2)  # parameters for tokenization, stopwords can be passed
    tfidf = tfidf_vectorizer.fit_transform(mes_train)
    tfidf_test = tfidf_vectorizer.transform(mes_test)

    ti2 = tfidf.T.A
    messages_tfidf = list(map(list, zip(*ti2)))

    ti3 = tfidf_test.T.A
    tfidf_test = list(map(list, zip(*ti3)))

    # print("TF-IDF feature names: ", tfidf_vectorizer.get_feature_names())
    # print("Number of TF-IDF features: ", len(tfidf_vectorizer.get_feature_names()))

    x_train = messages_tfidf
    x_test = tfidf_test

    return x_train, x_test, tfidf_vectorizer
