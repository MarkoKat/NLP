import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from similarity_analyse_crew import get_response_tfidf_dict, get_tfidf_books, get_book_dict


def use_similarities(response, book, vect, x_train, x_test, pred_train, pred_test,
                     class_train, class_test, book_idx_train, book_idx_test, response_link_train, response_link_test):

    if book:
        tfidf_books = get_tfidf_books(vect)
        book_dict = get_book_dict()

        book_idx_train = np.array(book_idx_train)
        book_idx_test = np.array(book_idx_test)

    if response:
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

    clf_rf = RandomForestClassifier(max_depth=10, random_state=0)

    # Train random forest
    clf_rf.fit(rf_arr_train, class_train)

    # Make predictions
    rf_pred = clf_rf.predict(rf_arr_test)

    # Evaluation
    print('Classification report - Random forest ---------------------------')
    print(metrics.classification_report(class_test, rf_pred, digits=3))
