import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from similarity_analyse_crew import get_response_tfidf_dict, get_tfidf_books, get_book_dict
from similarity_bert import get_response_bert_dict, get_bert_books
from sentence_transformers import SentenceTransformer
from confusion_matrix import get_confusion_matrix


def use_similarities(response, book, vect, x_train, x_test, pred_train, pred_test, class_train, class_test,
                     book_idx_train, book_idx_test, response_link_train, response_link_test, use_bert, class_names):

    bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

    if book:
        if use_bert:
            tfidf_books = get_bert_books(bert_model)
            book_dict = get_book_dict()
        else:
            tfidf_books = get_tfidf_books(vect)
            book_dict = get_book_dict()

        book_idx_train = np.array(book_idx_train)
        book_idx_test = np.array(book_idx_test)

    if response:
        if use_bert:
            response_tfidf_dict = get_response_bert_dict(bert_model)
        else:
            response_tfidf_dict = get_response_tfidf_dict(vect)

        response_link_train = np.array(response_link_train)
        response_link_test = np.array(response_link_test)

    rf_arr_train = []
    for i in range(len(pred_train)):
        if response and book:
            similarity_response = cosine_similarity([x_train[i]], [response_tfidf_dict[response_link_train[i]]])[0][0]
            similarity_book = cosine_similarity([x_train[i]], [tfidf_books[book_dict[book_idx_train[i]]]])[0][0]
            rf_arr_train.append([pred_train[i], similarity_response, similarity_book])

        if response and not book:
            similarity_response = cosine_similarity([x_train[i]], [response_tfidf_dict[response_link_train[i]]])[0][0]
            rf_arr_train.append([pred_train[i], similarity_response])

        if book and not response:
            similarity_book = cosine_similarity([x_train[i]], [tfidf_books[book_dict[book_idx_train[i]]]])[0][0]
            rf_arr_train.append([pred_train[i], similarity_book])

        # prob_list = prob_train[i].tolist()
        # prob_list.append(similarity_response)
        # prob_list.append(similarity_book)
        # rf_arr_train.append(prob_list)

    # print(rf_arr_train)

    rf_arr_test = []
    for i in range(len(pred_test)):
        if response and book:
            similarity_response = cosine_similarity([x_test[i]], [response_tfidf_dict[response_link_test[i]]])[0][0]
            similarity_book = cosine_similarity([x_test[i]], [tfidf_books[book_dict[book_idx_test[i]]]])[0][0]
            rf_arr_test.append([pred_test[i], similarity_response, similarity_book])

        if response and not book:
            similarity_response = cosine_similarity([x_test[i]], [response_tfidf_dict[response_link_test[i]]])[0][0]
            rf_arr_test.append([pred_test[i], similarity_response])

        if book and not response:
            similarity_book = cosine_similarity([x_test[i]], [tfidf_books[book_dict[book_idx_test[i]]]])[0][0]
            rf_arr_test.append([pred_test[i], similarity_book])

        # prob_list = prob_test[i].tolist()
        # prob_list.append(similarity_response)
        # prob_list.append(similarity_book)
        # rf_arr_test.append(prob_list)

    # print(rf_arr_test)

    clf_rf = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=10)

    # Train random forest
    clf_rf.fit(rf_arr_train, class_train)

    # Make predictions
    rf_pred = clf_rf.predict(rf_arr_test)

    # Evaluation
    print('Classification report - similarities were used ---------------------------')
    print(metrics.classification_report(class_test, rf_pred, digits=3, zero_division=0))

    get_confusion_matrix(class_test, rf_pred, class_names)
