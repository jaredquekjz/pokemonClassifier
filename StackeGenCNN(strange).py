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
from keras.models import Model

datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True, validation_split=0.2, rescale=1.0/255.0)

train_data = datagen.flow_from_directory(
    'pokemonData/train', batch_size=20, target_size=(128, 128), class_mode='categorical', subset='training')
test_data = datagen.flow_from_directory(
    'pokemonData/train', batch_size=20, target_size=(128, 128), class_mode='categorical', subset='validation')

# trainX = train_data[0][0]
# trainY = train_data[0][1]
# testX = test_data[0][0]
# testY = test_data[0][1]

# # Split out trainX, trainY, testX, testY
# for i in range(len(train_data)-1):
#     trainX = np.concatenate((trainX, train_data[i][0]))
#     trainY = np.concatenate((trainY, train_data[i][1]))
# for i in range(len(test_data)-1):
#     testX = np.concatenate((testX, test_data[i][0]))
#     testY = np.concatenate((testY, test_data[i][1]))


# Load models
# Convolutional Neural Net(1-block VGG)
def make_yolov3_model():
    input_image = Input(shape=(None, None, 3))
# Layer  0 => 4
    x = _conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                                  {'filter': 64, 'kernel': 3, 'stride': 2,
                                      'bnorm': True, 'leaky': True, 'layer_idx': 1},
                                  {'filter': 32, 'kernel': 1, 'stride': 1,
                                      'bnorm': True, 'leaky': True, 'layer_idx': 2},
                                  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])
# Layer  5 => 8
    x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                        {'filter':  64, 'kernel': 1, 'stride': 1,
                            'bnorm': True, 'leaky': True, 'layer_idx': 6},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])
# Layer  9 => 11
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])
# Layer 12 => 15
    x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 128, 'kernel': 1, 'stride': 1,
                            'bnorm': True, 'leaky': True, 'layer_idx': 13},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])
# Layer 16 => 36
    for i in range(7):
        x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])

    skip_36 = x

    # Layer 37 => 40
    x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                        {'filter': 256, 'kernel': 1, 'stride': 1,
                            'bnorm': True, 'leaky': True, 'layer_idx': 38},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])
# Layer 41 => 61
    for i in range(7):
        x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])

    skip_61 = x

    # Layer 62 => 65
    x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                        {'filter':  512, 'kernel': 1, 'stride': 1,
                            'bnorm': True, 'leaky': True, 'layer_idx': 63},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])
# Layer 66 => 74
    for i in range(3):
        x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])

    # Layer 75 => 79
    x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1,
                            'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter':  512, 'kernel': 1, 'stride': 1,
                            'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1,
                            'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False)
# Layer 80 => 82
    yolo_82 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
                              {'filter':  255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], skip=False)
# Layer 83 => 86
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1,
                         'bnorm': True, 'leaky': True, 'layer_idx': 84}], skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])
# Layer 87 => 91
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                        {'filter': 512, 'kernel': 3, 'stride': 1,
                            'bnorm': True, 'leaky': True, 'layer_idx': 88},
                        {'filter': 256, 'kernel': 1, 'stride': 1,
                            'bnorm': True, 'leaky': True, 'layer_idx': 89},
                        {'filter': 512, 'kernel': 3, 'stride': 1,
                            'bnorm': True, 'leaky': True, 'layer_idx': 90},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], skip=False)
# Layer 92 => 94
    yolo_94 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
                              {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], skip=False)
# Layer 95 => 98
    x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1,
                         'bnorm': True, 'leaky': True,   'layer_idx': 96}], skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])
# Layer 99 => 106
    yolo_106 = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
                               {'filter': 256, 'kernel': 3, 'stride': 1,
                                   'bnorm': True,  'leaky': True,  'layer_idx': 100},
                               {'filter': 128, 'kernel': 1, 'stride': 1,
                                   'bnorm': True,  'leaky': True,  'layer_idx': 101},
                               {'filter': 256, 'kernel': 3, 'stride': 1,
                                   'bnorm': True,  'leaky': True,  'layer_idx': 102},
                               {'filter': 128, 'kernel': 1, 'stride': 1,
                                   'bnorm': True,  'leaky': True,  'layer_idx': 103},
                               {'filter': 256, 'kernel': 3, 'stride': 1,
                                   'bnorm': True,  'leaky': True,  'layer_idx': 104},
                               {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], skip=False)

    model = Model(input_image, [yolo_82, yolo_94, yolo_106])
    return model


def define_model1():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     input_shape=(128, 128, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(18, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=1e-4),
                  metrics=['accuracy'])
    return model


def define_model2():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same', input_shape=(128, 128, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(18, activation='softmax'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# makedirs('models')
n_members = 5

# # simple early stopping
# #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

for i in range(n_members):
    model = define_model()
    history = model.fit_generator(train_data,
                                  steps_per_epoch=100,
                                  epochs=20,
                                  validation_data=test_data,
                                  validation_steps=50)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'g', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    print('Models' + str(i+1) + ": " + str(loss))
    filename = 'models/pokemonmodel_' + str(i+1) + '.h5'
    model.save(filename)
    print('>Saved %s' % filename)
    test_loss, test_acc = model.evaluate_generator(test_data, steps=24)
    print('Model Accuracy: %.3f' % test_acc)

# load models from file


def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        filename = 'models/pokemonmodel_' + str(i+1) + '.h5'
        model = load_model(filename)
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# # # %%
# # # create stacked model input dataset as outputs from the ensemble


# def stacked_dataset(members, inputX):
#     stackX = None
#     for model in members:
#         yhat = model.predict(testX)
#         # stack predictions into [rows, members, probabilities]
#         if stackX is None:
#             stackX = yhat
#         else:
#             stackX = dstack((stackX, yhat))
#     # flatten predictions

#     stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
#     return stackX


# # # %%
# def fit_stacked_model(members, inputX, inputy):

#     stackedX = stacked_dataset(members, inputX)
#     model = MLPClassifier(max_iter=1000)
#     model.fit(stackedX, inputy)
#     return model


# def stacked_prediction(members, model, inputX):
#     stackedX = stacked_dataset(members, inputX)
#     yhat = model.predict(stackedX)
#     return yhat


# n_members = 5
# members = load_all_models(n_members)
# print('Loaded %d models' % len(members))

# # Test accuracy of loaded models on test dataset
# for model in members:
#     test_loss, test_acc = model.evaluate_generator(test_data, steps=24)
#     print('Model Accuracy: %.3f' % test_acc)

# # fit stacked model using ensemble
# model = fit_stacked_model(members, testX, testY)
# # evaluate model on test set
# yhat = stacked_prediction(members, model, testX)
# acc = accuracy_score(testY, yhat)
# print()
# print('Stacked Test Accuracy: %.3f' % acc)


# # testX = testX.reshape(
# #     (testX.shape[0], testX.shape[1]*testX.shape[2]*testX.shape[3]))
# # trainX = trainX.reshape(
# #     (trainX.shape[0], trainX.shape[1]*trainX.shape[2]*trainX.shape[3]))

# # kfold = KFold(n_splits=10, random_state=7)
# # model = xgboost.XGBClassifier()
# # scoring = 'accuracy'
# # results = cross_val_score(model, trainX, trainY, cv=kfold, scoring=scoring)
# # print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))
