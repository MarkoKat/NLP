from torch import cuda
import pandas as pd
import numpy as np
import torch
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import f1_score

from DistillBERT import get_class_dict, get_message_classes, remove_small_classes
import random


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')



def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
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
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals


if __name__ == "__main__":
  print("Run BERT exp")

  device = 'cuda' if cuda.is_available() else 'cpu'
  # device = 'cpu'
  print("Device: ", device)

  file_name = 'imapbook.xlsx'
  # sheet_name = 'CREW data'
  sheet_name = "Discussion only data"

  df = pd.read_excel(file_name, sheet_name=sheet_name)
  # df.head()
  # # Removing unwanted columns and only leaving title of news and the category which will be the target
  df = df[['Message','CodePreliminary']]
  print(df.head())

  print("Prepare dataset")
  df_data = df

  # column Message
  messages = df_data['Message']

  # column CodePreliminary
  classes = df_data['CodePreliminary']

  class_dict, crew_dict_s = get_class_dict(classes)
  message_classes = get_message_classes(classes, class_dict)
  messages, message_classes = remove_small_classes(messages, message_classes, 5, crew_dict_s)

  sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=10)

  message_classes_np = np.array(message_classes)
  for train_index, test_index in sss.split(messages, message_classes):
      # print("TRAIN:", train_index, "TEST:", test_index)
      mes_train, mes_test = messages[train_index], messages[test_index]
      class_train, class_test = message_classes_np[train_index], message_classes_np[test_index]

  # intialise data of lists.
  data_train = {'Title':mes_train,
                'label':class_train}

  data_test = {'Title':mes_test,
              'label':class_test}
    
  # Create DataFrame
  df_train = pd.DataFrame(data_train)
  df_test = pd.DataFrame(data_test)

  # Creating the dataset and dataloader for the neural network
  train_dataset = df_train
  test_dataset = df_test
  if len(test_dataset) % 2 != 0:
    test_dataset = test_dataset[:-1]

  print("TRAIN Dataset: {}".format(train_dataset.shape))
  print("TEST Dataset: {}".format(test_dataset.shape))

  print(train_dataset.head())

  print("----------")

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)

  encoded_data_train = tokenizer.batch_encode_plus(
      train_dataset.Title.values, 
      add_special_tokens=True, 
      return_attention_mask=True, 
      pad_to_max_length=True, 
      max_length=256, 
      return_tensors='pt'
  )

  encoded_data_val = tokenizer.batch_encode_plus(
      test_dataset.Title.values, 
      add_special_tokens=True, 
      return_attention_mask=True, 
      pad_to_max_length=True, 
      max_length=256, 
      return_tensors='pt'
  )


  input_ids_train = encoded_data_train['input_ids']
  attention_masks_train = encoded_data_train['attention_mask']
  labels_train = torch.tensor(train_dataset.label.values)

  input_ids_val = encoded_data_val['input_ids']
  attention_masks_val = encoded_data_val['attention_mask']
  labels_val = torch.tensor(test_dataset.label.values)

  dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
  dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

  print(len(dataset_train), " - ", len(dataset_val))

  # -------

  label_dict = class_dict


  batch_size = 3

  dataloader_train = DataLoader(dataset_train, 
                                sampler=RandomSampler(dataset_train), 
                                batch_size=batch_size)

  dataloader_validation = DataLoader(dataset_val, 
                                    sampler=SequentialSampler(dataset_val), 
                                    batch_size=batch_size)


  model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  print(device)

  model.load_state_dict(torch.load('diss_finetuned_BERT_epoch_7.model', map_location=torch.device('cpu')))

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
