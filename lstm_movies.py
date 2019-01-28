import numpy as np
import os
import time
import string
import sys
import json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import random as rn
import keras
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#All this for reproducibility
np.random.seed(1)
rn.seed(1)
tf.set_random_seed(1)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)
# Build the corpus and sequences
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk_stopw = stopwords.words('english')
sequenceLength = 200

def tokenize (text):        #   no punctuation & starts with a letter & between 2-15 characters in length
    tokens = [word.strip(string.punctuation) for word in RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(text)]
    return  [f.lower() for f in tokens if f and f.lower() not in nltk_stopw]

def getMovies():
    X, labels, labelToName  = [], [], { 0 : 'neg', 1: 'pos' }
    for dataset in ['train', 'test']:
        for classIndex, directory in enumerate(['neg', 'pos']):
            dirName = './data/' + dataset + "/" + directory
            for reviewFile in os.listdir(dirName):
                with open (dirName + '/' + reviewFile, 'r') as f:
                    tokens = tokenize (f.read())
                    if (len(tokens) == 0):
                        continue
                X.append(tokens)
                labels.append(classIndex)
    nTokens = [len(x) for x in X]
    return X, np.array(labels), labelToName, nTokens

X, labels, labelToName, nTokens = getMovies()
print ('Token Summary:min/avg/median/std 85/86/87/88/89/90/95/99/max:',)
print (np.amin(nTokens), np.mean(nTokens),np.median(nTokens),np.std(nTokens),np.percentile(nTokens,85),np.percentile(nTokens,86),np.percentile(nTokens,87),np.percentile(nTokens,88),np.percentile(nTokens,89),np.percentile(nTokens,90),np.percentile(nTokens,95),np.percentile(nTokens,99),np.amax(nTokens))
labelToNameSortedByLabel = sorted(labelToName.items(), key=lambda kv: kv[0]) # List of tuples sorted by the label number [ (0, ''), (1, ''), .. ]
namesInLabelOrder = [item[1] for item in labelToNameSortedByLabel]
numClasses = len(namesInLabelOrder)
print ('X, labels #classes classes {} {} {} {}'.format(len(X), str(labels.shape), numClasses, namesInLabelOrder))

kTokenizer = keras.preprocessing.text.Tokenizer()
kTokenizer.fit_on_texts(X)
encoded_docs = kTokenizer.texts_to_sequences(X)
Xencoded = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=sequenceLength, padding='post')
print ('Vocab padded_docs {} {}'.format(len(kTokenizer.word_index), str(Xencoded.shape)))
# Test & Train Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(Xencoded, labels)
train_indices, test_indices = next(sss)
train_x, test_x = Xencoded[train_indices], Xencoded[test_indices]
train_labels = keras.utils.to_categorical(labels[train_indices], len(labelToName))
test_labels = keras.utils.to_categorical(labels[test_indices], len(labelToName))

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2, mode='auto', restore_best_weights=False)
model = keras.models.Sequential()
embedding = keras.layers.embeddings.Embedding(input_dim=len(kTokenizer.word_index)+1, output_dim=300, input_length=sequenceLength, trainable=True, mask_zero=True)
model.add(embedding)
model.add(keras.layers.LSTM(units=150, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
model.add(keras.layers.Dense(numClasses, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

start_time = time.time()
result = {}
history = model.fit(x=train_x, y=train_labels, epochs=50, batch_size=32, shuffle=True, validation_data = (test_x, test_labels), verbose=2, callbacks=[early_stop])
result['history'] = history.history
result['test_loss'], result['test_accuracy'] = model.evaluate(test_x, test_labels, verbose=2)
predicted = model.predict(test_x, verbose=2)
predicted_labels = predicted.argmax(axis=1)
result['confusion_matrix'] = confusion_matrix(labels[test_indices], predicted_labels).tolist()
result['classification_report'] = classification_report(labels[test_indices], predicted_labels, digits=4, target_names=namesInLabelOrder, output_dict=True)
print (confusion_matrix(labels[test_indices], predicted_labels))
print (classification_report(labels[test_indices], predicted_labels, digits=4, target_names=namesInLabelOrder))
elapsed_time = time.time() - start_time
print ('Time Taken:', elapsed_time)
result['elapsed_time'] = elapsed_time

f = open ('lstm.json','w')
out = json.dumps(result, ensure_ascii=True)
f.write(out)
f.close()

