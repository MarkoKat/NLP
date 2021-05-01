from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    texts = ["My car is really fast",
             "I like driving cars",
             "This flower are beautiful",
             "Taking care of flowers is realxing",
             "Cars are my hobby",
             "Flowers are my hobby"]

    print('TF-IDF feature names:')

    # vect_b = TfidfVectorizer(ngram_range=(1, 2))
    vect_b = TfidfVectorizer(min_df=3)
    tfidf_messages_b = vect_b.fit_transform(texts)
    print(vect_b.get_feature_names())
