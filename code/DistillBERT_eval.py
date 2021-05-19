import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
from sentence_transformers import SentenceTransformer

from torch import cuda
from sklearn import metrics
import sys
import time

from prepare_data import get_data
from DistillBERT import Triage, DistillBERTClass, calcuate_accu
from similarities import use_similarities
from tfidf_helper import get_tfidf_vectors
from confusion_matrix import get_confusion_matrix

if __name__ == "__main__":
    print("--- DistilBERT ---")

    device = 'cuda' if cuda.is_available() else 'cpu'
    print("Device: ", device)

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

    model_name = arguments[1]
    # ---

    PATH = model_name

    # model = DistillBERTClass()
    model = torch.load(PATH)
    model.to(device)
    model.eval()

    MAX_LEN = 128
    VALID_BATCH_SIZE = 2
    EPOCHS = 1
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

    # Get data
    mes_train, mes_test, class_train, class_test, book_idx_train, book_idx_test, response_link_train, response_link_test, class_dict = get_data(sheet, use_response_similarity, use_book_similarity)

    # intialise data of lists.
    data_test = {'Message': mes_test,
                 'ENCODE_CAT': class_test}

    # Create DataFrame
    df_test = pd.DataFrame(data_test)

    # Creating the dataset and dataloader for the neural network
    test_dataset = df_test
    if len(test_dataset) % 2 != 0:
        test_dataset = test_dataset[:-1]

    # print("TEST Dataset: {}".format(test_dataset.shape))

    # Creating the loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # Predictions - Evalvacija
    start_time = time.time()
    testing_set = Triage(test_dataset, tokenizer, MAX_LEN)

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': 0}

    testing_loader = DataLoader(testing_set, **test_params)

    true_classes = []
    predicted_classes = []

    n_correct = 0
    n_wrong = 0
    total = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask).squeeze()
            # print(outputs)
            # print(targets)
            targets_list = targets.tolist()
            true_classes.append(targets_list[0])
            true_classes.append(targets_list[1])

            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            # print(big_idx)
            big_idx_list = big_idx.tolist()
            predicted_classes.append(big_idx_list[0])
            predicted_classes.append(big_idx_list[1])

            n_correct += calcuate_accu(big_idx, targets)
            # print(calcuate_accu(big_idx, targets))

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

    print("--- Evaluation time: %s seconds ---" % (time.time() - start_time))

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples

    # print(true_classes)
    # print(predicted_classes)

    # print(f"Validation Loss Epoch: {epoch_loss}")
    # print(f"Test Accuracy: {epoch_accu}")

    print('Classification report (DistilBERT) ---------------------------')
    print(metrics.classification_report(true_classes, predicted_classes, digits=3, zero_division=0))

    # Confusion matrix
    class_names = [class_dict[x] for x in list(set(class_test))]
    get_confusion_matrix(true_classes, predicted_classes, class_names)

    # --- Similarities -----------------------------------------------------------
    if use_response_similarity or use_book_similarity:
        print("--- SIMILARITIES ---")

        pred_train = class_train
        pred_test = predicted_classes
        tfidf_vectorizer = None

        if len(class_test) % 2 != 0:
            class_test = class_test[:-1]

        if use_bert:
            print("Use BERT embeddings for similarity")
            bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
            x_train = bert_model.encode(mes_train)
            x_test = bert_model.encode(mes_test)
        else:
            print("Use TF-IDF vectors for similarity")
            x_train, x_test, tfidf_vectorizer = get_tfidf_vectors(mes_train, mes_test)

        use_similarities(use_response_similarity, use_book_similarity, tfidf_vectorizer, x_train, x_test,
                         pred_train, pred_test, class_train, class_test,
                         book_idx_train, book_idx_test, response_link_train, response_link_test, use_bert, class_names)
