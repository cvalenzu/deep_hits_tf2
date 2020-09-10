from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime

from model import model
from tensorflow import keras
from tensorflow.keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


with open("X_clean.npy", "rb+") as f:
    X = np.load(f)

with open("y_clean.npy", "rb+") as f:
    y = np.load(f)

le = LabelEncoder()
le.fit(y)
y = le.transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)


with open("X_test.npy", "wb+") as f:
    np.save(f,X_test)

with open("y_test.npy", "wb+") as f:
    np.save(f,y_test)

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


early_callback = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=1e-5,patience=5)
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath="best_so_far.h5",save_best_only=True)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy","categorical_accuracy",f1])
model.fit(X_train,y_train,validation_split=0.2,batch_size=500,epochs=1000,callbacks=[tensorboard_callback,early_callback,checkpoint_callback])
model.save("model.h5")
