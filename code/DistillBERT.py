# Importing the libraries needed
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer

from torch import cuda
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics


def encode_cat(x, encode_dict):
    x = x.lower()
    if x[-1] == " ":
      x = x[:-1]
    if x not in encode_dict.keys():
        encode_dict[x]=len(encode_dict)
        # print(x)
    return encode_dict[x]


class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        title = str(self.data.Message[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.ENCODE_CAT[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len


def get_class_dict(classes_f):
    """Get dictionary for string and numerical form of classes"""
    class_dict = {}
    index = 0
    for code in classes_f:
        code = code.lower()
        if code[-1] == " ":
            code = code[:-1]
        if code not in class_dict:
            class_dict[code] = index
            index += 1

    print('Classes:')
    print(class_dict)

    # Switch keys and values in dict
    crew_dict_s = {y: x for x, y in class_dict.items()}
    # print(crew_dict_s)

    return class_dict, crew_dict_s


def get_message_classes(classes_f, class_dict):
    """Get array with class for each message in numerical form"""
    message_classes = []

    for el in classes_f:
        el = el.lower()
        if el[-1] == " ":
            el = el[:-1]
        message_classes.append(class_dict[el])
    return message_classes


def remove_small_classes(messages, message_classes, min_number_of_messages, crew_dict_s):
    """Remove messages from classes with number of instances smaller than min_number_of_messages"""

    all_dict = {}
    for code in message_classes:
        if crew_dict_s[code] not in all_dict:
            all_dict[crew_dict_s[code]] = 1
        else:
            all_dict[crew_dict_s[code]] += 1

    # all_dict_print = collections.OrderedDict(sorted(all_dict.items()))
    # print("Class counts")
    # print(all_dict_print)

    all_dict_copy = all_dict.copy()
    del_keys = []
    for key in all_dict_copy:
        if all_dict_copy[key] < min_number_of_messages:
            del all_dict[key]
            del_keys.append(key)

    print('Delete keys list: ', del_keys)

    print("Data length: ", len(message_classes), " - ", len(messages))

    messages_np = np.array(messages)
    for i in range(len(message_classes)):
        for del_key in del_keys:
            if crew_dict_s[message_classes[i]] == del_key:
                # print(del_key)
                # print(crew_messages[i])
                message_classes[i] = None
                messages_np[i] = None
                break

    message_classes = list(filter(lambda a: a is not None, message_classes))
    messages = list(filter(lambda a: a is not None, messages_np))
    messages = np.array(messages)
    print("Data length (after removal): ", len(message_classes), " - ", len(messages))
    return messages, message_classes


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 
class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        # self.classifier = torch.nn.Linear(768, 16)
        self.classifier = torch.nn.Linear(768, 9)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


# Function to calcuate the accuracy of the model
def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct


# Defining the training function on the 80% of the dataset for tuning the distilbert model
def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        # if _%5000==0:
        #     loss_step = tr_loss/nb_tr_steps
        #     accu_step = (n_correct*100)/nb_tr_examples 
        #     print(f"Training Loss per 5000 steps: {loss_step}")
        #     print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return 


# ------------------------------------------------------------------------------
if __name__ == "__main__":
  print("Run DistillBERT")

  device = 'cuda' if cuda.is_available() else 'cpu'
  # device = 'cpu'
  print("Device: ", device)

  file_name = '..\\data\\Popravki - IMapBook - CREW and discussions dataset.xlsx'
  # sheet_name = 'CREW data'
  sheet_name = "Discussion only data"

  df = pd.read_excel(file_name, sheet_name=sheet_name)
  # df.head()
  # # Removing unwanted columns and only leaving title of news and the category which will be the target
  df = df[['Message','CodePreliminary']]
  # print(df.head())

  encode_dict = {}
  df['ENCODE_CAT'] = df['CodePreliminary'].apply(lambda x: encode_cat(x, encode_dict))

  # Defining some key variables that will be used later on in the training
  MAX_LEN = 128
  TRAIN_BATCH_SIZE = 4
  VALID_BATCH_SIZE = 2
  EPOCHS = 1
  LEARNING_RATE = 1e-05
  print("DistilBertTokenizer")
  tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

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

  model = DistillBERTClass()
  model.to(device)

  # Creating the loss function and optimizer
  loss_function = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

  # Train the model
  for epoch in range(16):
    train(epoch)

  # Evalvacija
  true_classes = []
  predicted_classes = []

  model.eval()
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
          try:
            true_classes.append(targets_list[1])
          except:
            print("Batch with one element")

          loss = loss_function(outputs, targets)
          tr_loss += loss.item()
          big_val, big_idx = torch.max(outputs.data, dim=1)
          # print(big_idx)
          big_idx_list = big_idx.tolist()
          predicted_classes.append(big_idx_list[0])
          try:
            predicted_classes.append(big_idx_list[1])
          except:
            print("Batch with one element")

          n_correct += calcuate_accu(big_idx, targets)
          # print(calcuate_accu(big_idx, targets))

          nb_tr_steps += 1
          nb_tr_examples+=targets.size(0)
          
          # if _%5000==0:
          #     loss_step = tr_loss/nb_tr_steps
          #     accu_step = (n_correct*100)/nb_tr_examples
          #     print(f"Validation Loss per 100 steps: {loss_step}")
          #     print(f"Validation Accuracy per 100 steps: {accu_step}")
  epoch_loss = tr_loss/nb_tr_steps
  epoch_accu = (n_correct*100)/nb_tr_examples

  print(true_classes)
  print(predicted_classes)

  print(f"Validation Loss Epoch: {epoch_loss}")
  print(f"Validation Accuracy Epoch: {epoch_accu}")

  print(metrics.classification_report(true_classes, predicted_classes, digits=3))

  # Saving the files for re-use
  output_model_file = 'distill_bert_diss_2.bin'
  model_to_save = model
  torch.save(model_to_save, output_model_file)
