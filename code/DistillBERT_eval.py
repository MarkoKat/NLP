import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer

from torch import cuda
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics

from DistillBERT import DistillBERTClass, encode_cat, get_class_dict, get_message_classes, remove_small_classes, Triage, calcuate_accu


if __name__ == "__main__":
  print("Eval DistillBERT")

  device = 'cuda' if cuda.is_available() else 'cpu'
  print("Device: ", device)

  PATH = "distill_bert_diss_1.bin"

  # model = DistillBERTClass()
  model = torch.load(PATH)
  model.to(device)
  model.eval()

  # Load test dataset
  file_name = 'imapbook.xlsx'
  # sheet_name = 'CREW data'
  sheet_name = "Discussion only data"

  df = pd.read_excel(file_name, sheet_name=sheet_name)
  df = df[['Message','CodePreliminary']]

  encode_dict = {}
  df['ENCODE_CAT'] = df['CodePreliminary'].apply(lambda x: encode_cat(x, encode_dict))

  MAX_LEN = 128
  TRAIN_BATCH_SIZE = 4
  VALID_BATCH_SIZE = 2
  EPOCHS = 1
  LEARNING_RATE = 1e-05
  tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

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
  data_train = {'Message':mes_train,
                'ENCODE_CAT':class_train}

  data_test = {'Message':mes_test,
              'ENCODE_CAT':class_test}

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

  training_set = Triage(train_dataset, tokenizer, MAX_LEN)
  testing_set = Triage(test_dataset, tokenizer, MAX_LEN)

  train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

  test_params = {'batch_size': VALID_BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 0
                  }

  training_loader = DataLoader(training_set, **train_params)
  testing_loader = DataLoader(testing_set, **test_params)

  # Creating the loss function and optimizer
  loss_function = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

  # Evalvacija
  true_classes = []
  predicted_classes = []

  n_correct = 0; n_wrong = 0; total = 0
  tr_loss = 0
  nb_tr_steps = 0
  nb_tr_examples = 0
  with torch.no_grad():
      for _, data in enumerate(testing_loader, 0):
          ids = data['ids'].to(device, dtype = torch.long)
          mask = data['mask'].to(device, dtype = torch.long)
          targets = data['targets'].to(device, dtype = torch.long)
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
          nb_tr_examples+=targets.size(0)
          
  epoch_loss = tr_loss/nb_tr_steps
  epoch_accu = (n_correct*100)/nb_tr_examples

  print(true_classes)
  print(predicted_classes)

  # print(f"Validation Loss Epoch: {epoch_loss}")
  print(f"Validation Accuracy Epoch: {epoch_accu}")

  print(metrics.classification_report(true_classes, predicted_classes, digits=3))