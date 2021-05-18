# USAGE
# python keras_cifar10.py --output output/keras_cifar10.png

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer # one hot encoding for labels
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential # build NN architecture
from tensorflow.keras.layers import Dense # fully connected layer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# load the training and testing data, scale it into the range [0, 1],
# then reshape the design matrix
print("[INFO] loading CIFAR-10 data...")
# loads the data presplit into train/test
((trainX, trainY), (testX, testY)) = cifar10.load_data()
# scale 0, 1
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
# 32*32*3 (for each image there are 1024 pixels, but for each color RGB)
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
# this is for humman readable out
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
	"dog", "frog", "horse", "ship", "truck"]

# define the 3072-1024-512-10 architecture using Keras
model = Sequential()
# input shape is 3072, apply Dense function, relu act. f(x), output 1024
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
# input accepts 1024, outputs 512
model.add(Dense(512, activation="relu"))
# accepts 512, outputs 10
model.add(Dense(10, activation="softmax"))

# train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
# builds the model
model.compile(loss="categorical_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
# trains the model
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=100, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
# batch size, larger dataset == MORE MEMORY CONSUMED (e.g. imagenet moving data off RAM)
# larger the batch, fewer the weight updates there are
# smaller the batch, the longer the duration to complete, basically std gradient descent
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])