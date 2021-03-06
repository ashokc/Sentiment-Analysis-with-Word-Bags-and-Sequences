import matplotlib.pyplot as plt
import json
import itertools
import numpy as np

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, filename='cf-matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig = plt.figure(figsize=(6,6),dpi=720)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#    fig.align_labels()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    fig.savefig(filename+"-cf.png")

label_names = ['neg', 'pos']
with open ('./svm.json') as fh:
    result = json.loads(fh.read())
    cm = result['confusion_matrix']
    cm = np.array([np.array(xi) for xi in cm])
    plot_confusion_matrix(cm, classes=np.asarray(label_names), normalize=False, title='SVM', filename='svm')

with open ('./lstm.json') as fh:
    result = json.loads(fh.read())
    cm = result['confusion_matrix']
    cm = np.array([np.array(xi) for xi in cm])
    plot_confusion_matrix(cm, classes=np.asarray(label_names), normalize=False, title='LSTM', filename='lstm')

