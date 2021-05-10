import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sentence_transformers import SentenceTransformer
import sys

from stemming import get_stemms, get_lemmas
from stop_words import remove_stopwords
from similarity_analyse_crew import get_response_tfidf_dict, get_tfidf_books, get_book_dict
from similarities import use_similarities
from prepare_data import get_data


def get_predictions(clf_f, train_data, test_data):
    pred_train = clf_f.predict(train_data)
    pred_test = clf_f.predict(test_data)
    return pred_train, pred_test


def get_probabilities(clf_f, train_data, test_data):
    prob_train = clf_f.predict_proba(train_data)
    prob_test = clf_f.predict_proba(test_data)
    return prob_train, prob_test


# ======================================================================================================================
if __name__ == "__main__":

    # sheet = 'crew'
    sheet = 'discussion'

    use_response_similarity = False  # Can't use with discussion
    use_book_similarity = False

    use_bert = False

    # Read command line arguments
    arguments = sys.argv
    print("Arguments: ", arguments)

    if 'crew' in arguments:
        sheet = 'crew'
    if 'discussion' in arguments:
        sheet = 'discussion'
    if 'use_response_similarity' in arguments:
        use_response_similarity = True
    if 'use_book_similarity' in arguments:
        use_book_similarity = True
    if 'use_bert_for_similarity' in arguments:
        use_bert = True

    # Get data
    mes_train, mes_test, class_train, class_test, book_idx_train, book_idx_test, response_link_train, response_link_test = get_data(sheet, use_response_similarity, use_book_similarity)

    print("Train length: ", len(class_train), " Test length: ", len(class_test))

    # ------------------ TFIDF ----------------------------

    tfidf_vectorizer = TfidfVectorizer(min_df=2)  # parameters for tokenization, stopwords can be passed
    tfidf = tfidf_vectorizer.fit_transform(mes_train)
    tfidf_test = tfidf_vectorizer.transform(mes_test)

    ti2 = tfidf.T.A
    messages_tfidf = list(map(list, zip(*ti2)))

    ti3 = tfidf_test.T.A
    tfidf_test = list(map(list, zip(*ti3)))

    # print("TF-IDF feature names: ", tfidf_vectorizer.get_feature_names())
    print("Number of TF-IDF features: ", len(tfidf_vectorizer.get_feature_names()))

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

    # Train MLP
    clf.fit(x_train, class_train)
    print("Number of iterations: ", clf.n_iter_)

    # --- Make predictions ------------------------------------------------
    predictions = clf.predict(x_test)

    # print('Predictions:')
    # for p in range(len(predictions)):
    #     str_pred = crew_dict_s[predictions[p]]
    #     str_true = crew_dict_s[class_test[p]]
    #     print('Predicted: ', str_pred, ' - True: ', str_true)

    # --- Evaluation --------------------------------------------------

    # print('Confusion matrix --------------------------------')
    # print(metrics.confusion_matrix(class_test, predictions))

    print('Classification report ---------------------------')
    print(metrics.classification_report(class_test, predictions, digits=3))

    # =============================================================================================================

    # --- Similarities -----------------------------------------------------------
    if use_response_similarity or use_book_similarity:
        print("--- SIMILARITIES ---")

        pred_train, pred_test = get_predictions(clf, x_train, x_test)
        prob_train, prob_test = get_probabilities(clf, x_train, x_test)

        # pred_train = class_train

        # print(class_test)
        # print(pred_test)

        if use_bert:
            bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
            x_train = bert_model.encode(mes_train)
            x_test = bert_model.encode(mes_test)

        use_similarities(use_response_similarity, use_book_similarity, tfidf_vectorizer, x_train, x_test,
                         pred_train, pred_test, class_train, class_test,
                         book_idx_train, book_idx_test, response_link_train, response_link_test, use_bert)
