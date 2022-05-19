# Loading relevant packages
import os, sys
sys.path.append(os.path.join("..","..","CDS-VIS"))

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)

from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

import numpy as np
import matplotlib.pyplot as plt

# Importing Cifar10, normalizing and lableling
def import_Cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train/255
    X_test = X_test/255
    
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    label_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    
    return X_train, X_test, y_train, y_test, label_names

# loading VGG16
def import_VGG16():
    model = VGG16(include_top = False,
                  pooling = "avg",
                  input_shape = (32,32,3))
    
    return model
    
# Training Classifier
def classifier(model, X_train, y_train, X_test, y_test):
    for layer in model.layers:
        layer.trainable = False
    
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation="relu")(flat1)
    output = Dense(10, activation="softmax")(class1)
    
    model = Model(inputs = model.inputs,
                  outputs = output)
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)

    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=sgd,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    epochs = 10
    
    H = model.fit(X_train, y_train,
                  validation_data = (X_test, y_test),
                  batch_size = 128,
                  epochs = 10,
                  verbose = 1)
    
    return H, model, epochs

# Plotting and saving loss, accuracy and classification report
def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()

    outpath = os.path.join("out","loss_accuracy_plot_VGG16.jpg")
    
    plt.savefig(outpath)

def class_report(model, X_test, y_test, label_names):
    predictions = model.predict(X_test, batch_size=128)
    class_report = classification_report(y_test.argmax(axis=1), #printing classification
                                predictions.argmax(axis=1),
                                target_names = label_names)
    print(class_report)
    
    outpath = os.path.join("out","classification_report.txt")
    
    with open(outpath,"w") as file:
        file.write(str(class_report))

# Defining main
def main():
    X_train, X_test, y_train, y_test, label_names = import_Cifar10()
    model = import_VGG16()
    H, model, epochs = classifier(model, X_train, y_train, X_test, y_test)
    plot_history(H, epochs)
    class_report(model, X_test, y_test, label_names)
              
# Running main
if __name__ == "__main__":
    main()