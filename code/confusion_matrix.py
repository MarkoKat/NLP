from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


def get_confusion_matrix(true_classes, predictions, class_names):
    cm = metrics.confusion_matrix(true_classes, predictions)
    np.set_printoptions(precision=2)
    # print('Confusion matrix, without normalization ---------------')
    # print(cm)

    df_cm = pd.DataFrame(cm, class_names, class_names)
    sn.set(font_scale=0.7)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})  # font size

    plt.tight_layout()
    plt.show()
