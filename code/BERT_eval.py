from torch import cuda
import pandas as pd
import numpy as np
import torch
from sklearn import metrics
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score

from prepare_data import get_data


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def evaluate(dataloader_val):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


if __name__ == "__main__":
    print("Run BERT exp")

    device = 'cuda' if cuda.is_available() else 'cpu'
    # device = 'cpu'
    print("Device: ", device)

    sheet = 'crew'
    # sheet = 'discussion'

    use_response_similarity = False  # Can't use with discussion
    use_book_similarity = True

    # Get data
    mes_train, mes_test, class_train, class_test, book_idx_train, book_idx_test, response_link_train, response_link_test = get_data(
        sheet, use_response_similarity, use_book_similarity)

    NUM_CLASSES = 16
    if sheet == "discussion":
        NUM_CLASSES = 9
    # ---

    # intialise data of lists.

    data_test = {'Title': mes_test,
                 'label': class_test}

    # Create DataFrame
    df_test = pd.DataFrame(data_test)

    # Creating the dataset and dataloader for the neural network
    test_dataset = df_test
    if len(test_dataset) % 2 != 0:
        test_dataset = test_dataset[:-1]

    print("TEST Dataset: {}".format(test_dataset.shape))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)

    encoded_data_val = tokenizer.batch_encode_plus(
        test_dataset.Title.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(test_dataset.label.values)

    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    # -------

    batch_size = 3

    dataloader_validation = DataLoader(dataset_val,
                                       sampler=SequentialSampler(dataset_val),
                                       batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=NUM_CLASSES,
                                                          output_attentions=False,
                                                          output_hidden_states=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(device)

    model.load_state_dict(torch.load('finetuned_BERT_epoch_9.model', map_location=torch.device('cpu')))

    _, predictions, true_vals = evaluate(dataloader_validation)
    # print(predictions)
    idxs = []
    predictions = predictions.tolist()
    for el in predictions:
        idxs.append(el.index(max(el)))
    print(idxs)
    true_vals = true_vals.tolist()
    print(true_vals)

    print(metrics.classification_report(true_vals, idxs, digits=3))
