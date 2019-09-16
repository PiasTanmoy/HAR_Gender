# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 07:44:02 2019

@author: Pias Tanmoy
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:02:52 2019

@author: Pias Tanmoy
"""


from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import shuffle
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import glob
from sklearn.model_selection import train_test_split
import keras

np.random.seed(0)
N_EPOCH = 20
BATCH_SIZE = 50
VERBOSE = 1
N_CLASS = 2
OPTIMIZER = Adam()
N_HIDDEN_1 = 128
VALIDATION_SPLIT = 0.1
RESHAPE = 784
DROPOUT = 0.01



path = r'H:\HAR_Male_Female\HAR_Male_Female\dataset\male' # use your path
all_files = glob.glob(path + "/*.csv")

count = 0

X_M = pd.read_csv('H:\HAR_Male_Female\HAR_Male_Female\dataset\m_abubakar_180_01_1201011019.csv')
X_M = np.array(X_M)

Y = []
Y =[count for i in range(X_M.shape[0])]
Y = np.array(Y)
Y = Y.reshape(Y.shape[0],1)


for filename in all_files:
    df = pd.read_csv(filename)
    df = np.array(df)
    X_M = np.concatenate((X_M, df), axis=0)
    
    count = count + 1
    c = []
    c =[count for i in range(df.shape[0])]
    c = np.array(c)
    c = c.reshape(c.shape[0],1)
    
    Y = np.concatenate((Y, c), axis=0)
    
    print(df.shape)
    print(X_M.shape)
    print(Y.shape)
    print("\n")



path = r'H:\HAR_Male_Female\HAR_Male_Female\dataset\female' # use your path
all_files = glob.glob(path + "/*.csv")



for filename in all_files:
    df = pd.read_csv(filename)
    df = np.array(df)
    X_M = np.concatenate((X_M, df), axis=0)
    
    count = count + 1
    c = []
    c =[count for i in range(df.shape[0])]
    c = np.array(c)
    c = c.reshape(c.shape[0],1)
    
    Y = np.concatenate((Y, c), axis=0)
    
    
    print(df.shape)
    print(X_M.shape)
    print(Y.shape)
    print("\n")







X = X_M
np.savetxt("X.csv", X, delimiter=",")
np.savetxt("Y_Individual.csv", Y, delimiter=",")

Y = pd.read_csv("Y_Individual.csv")
Y = np.array(Y)

onehotencoder = OneHotEncoder(categorical_features = [0])
Y = onehotencoder.fit_transform(Y).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


#X_train, Y_train = shuffle(X_train, y_train)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def create_model():
  classifier = Sequential()
  classifier.add(Dense(units = 50, activation='relu', kernel_initializer='glorot_uniform', input_dim=6))
  classifier.add(Dropout(DROPOUT))
  classifier.add(Dense(units = 200, activation='relu', kernel_initializer = 'glorot_uniform'))
  classifier.add(Dropout(DROPOUT))
  classifier.add(Dense(units = 200, activation='relu', kernel_initializer = 'glorot_uniform'))
  classifier.add(Dropout(DROPOUT))
  classifier.add(Dense(units = 200, activation='relu', kernel_initializer = 'glorot_uniform'))
  classifier.add(Dropout(DROPOUT))
  classifier.add(Dense(units = 200, activation='relu', kernel_initializer = 'glorot_uniform'))
  classifier.add(Dropout(DROPOUT))
  classifier.add(Dense(units = 50, activation='relu', kernel_initializer = 'glorot_uniform'))
  classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'sigmoid'))
  classifier.compile(optimizer = 'Adamax' , loss = 'categorical_crossentropy', metrics = ['accuracy'])
  model = classifier
  
  return model

model = create_model()

#model = create_model()

BATCH_SIZE = 200

history = model.fit(X_train, y_train, batch_size = BATCH_SIZE, 
                    epochs = 20, verbose = VERBOSE, 
                    validation_split=VALIDATION_SPLIT,
                    shuffle = True)



scores = model.evaluate(X_test, y_test, verbose=1)
print("Test Score: ", scores[0])
print("Accuracy: " , scores[1])

y_pred = model.predict(X_test)
y_test_argmax = y_test.argmax(axis=1)
y_pred_argmax = y_pred.argmax(axis=1)

from sklearn.metrics import f1_score
print('sklearn Macro-F1-Score:', f1_score(y_test_argmax, y_pred_argmax, average='macro'))

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
y_test_argmax = y_test.argmax(axis=1)
y_pred_argmax = y_pred.argmax(axis=1)

class_names = np.array([0, 1, 2, 3, 4, 5])

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test_argmax, y_pred_argmax, classes=class_names,
                      title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plot_confusion_matrix(y_test_argmax, y_pred_argmax, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

