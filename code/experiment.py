from sklearn.model_selection import StratifiedShuffleSplit


if __name__ == "__main__":
    messages = ['ayy', 'nee', 'neki', 'buu', 'drevo', 'cesta']
    message_classes = [1, 1, 2, 2, 2, 2]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=10)

    for train_index, test_index in sss.split(messages, message_classes):
        print("TRAIN:", train_index, "TEST:", test_index)