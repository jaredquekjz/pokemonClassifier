
# Pokemon classification deep neural net. Classifies Pokemons into 10 different classes.

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from os import makedirs
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from keras.models import load_model
from numpy import dstack
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import RMSprop, SGD
from keras.callbacks.callbacks import EarlyStopping
from sklearn.model_selection import KFold, cross_val_score
import xgboost
from matplotlib import pyplot as plt
from os import path
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications import VGG16
from sklearn.externals import joblib

# Image loading and preprocessing

datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True, validation_split=0.2, rescale=1.0/255.0)

train_data = datagen.flow_from_directory(
    'pokemonProject/train', batch_size=20, target_size=(150, 150), class_mode='categorical', subset='training')
test_data = datagen.flow_from_directory(
    'pokemonProject/train', batch_size=20, target_size=(150, 150), class_mode='categorical', subset='validation')

# Split out trainX, trainY, testX, testY (for stacked generalisation)

trainX = train_data[0][0]
trainY = train_data[0][1]
testX = test_data[0][0]
testY = test_data[0][1]

for i in range(len(train_data)-1):
    trainX = np.concatenate((trainX, train_data[i][0]))
    trainY = np.concatenate((trainY, train_data[i][1]))
for i in range(len(test_data)-1):
    testX = np.concatenate((testX, test_data[i][0]))
    testY = np.concatenate((testY, test_data[i][1]))


# Load 3 pre-trained conv layers (features) topped with new dense layers

def define_model(num):

    if num == 0:
        convbase = MobileNetV2(
            weights='imagenet', include_top=False, input_shape=(150, 150, 3))
        print("Creating MobileNet layer!")
    elif num == 1:
        convbase = ResNet50V2(weights='imagenet',
                              include_top=False, input_shape=(150, 150, 3))
        print("Creating ResNet layer!")
    elif num == 2:
        convbase = VGG16(
            weights='imagenet', include_top=False, input_shape=(150, 150, 3))
        print("Creating VGG16 layer!")
    else:
        print("Out of range! Something's wrong")

    convbase.trainable = False  # Preserve weights of pre-trained conv layers
    model = Sequential()
    model.add(convbase)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=2e-5),
                  metrics=['accuracy'])
    return model

# load models from file


def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        filename = 'models/pokemonmodel_' + str(i+1) + '.h5'
        model = load_model(filename)
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# # create stacked model input dataset as outputs from the ensemble of three models


def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        yhat = model.predict(testX)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
    # flatten predictions

    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX


def fit_stacked_model(members, inputX, inputy):

    stackedX = stacked_dataset(members, inputX)
    model = MLPClassifier(max_iter=1000)
    model.fit(stackedX, inputy)
    return model


def stacked_prediction(members, model, inputX):
    stackedX = stacked_dataset(members, inputX)
    yhat = model.predict(stackedX)
    return yhat


# main
if path.exists('models') == 0:
    makedirs('models')

n_members = 3  # Fitting and saving three different models

for i in range(n_members):
    model = define_model(i)
    history = model.fit_generator(train_data,
                                  steps_per_epoch=100,
                                  epochs=10,
                                  validation_data=test_data,
                                  validation_steps=50)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    # plt.plot(epochs, acc, 'g', label='Training acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()
    # plt.figure()
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    # plt.show()
    print('Models' + str(i+1) + ": " + str(loss))
    filename = 'models/pokemonmodel_' + str(i+1) + '.h5'
    model.save(filename)
    print('>Saved %s' % filename)
    test_loss, test_acc = model.evaluate_generator(test_data, steps=24)
    print('Model Accuracy: %.3f' % test_acc)


members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# Test accuracy of loaded models on test dataset
# for model in members:
#     test_loss, test_acc = model.evaluate_generator(test_data, steps=24)
#     print('Model' + str(i+1) + ' Accuracy:' + str('test_acc'))


# Stacked generalisation of three models
print('Producing final model...')
model = fit_stacked_model(members, testX, testY)

# evaluate stacked model on test set
yhat = stacked_prediction(members, model, testX)
acc = accuracy_score(testY, yhat)
print()
filename = 'models/finalpokemonmodel.sav'
joblib.dump(model, filename)
print('Final Pokemonmodel is complete and saved.')
print('Stacked Test Accuracy: %.3f' % acc)


# kfold = KFold(n_splits=10, random_state=7)
# model = xgboost.XGBClassifier()
# scoring = 'accuracy'
# results = cross_val_score(model, trainX, trainY, cv=kfold, scoring=scoring)
# print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))