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
from keras.optimizers import SGD
from sklearn.metrics import precision_score
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score


datagen = ImageDataGenerator(
    horizontal_flip=True, validation_split=0.2, rescale=1.0/255.0)

train_data = datagen.flow_from_directory(
    'pokemonData/train', batch_size=1, class_mode='categorical', subset='training')
test_data = datagen.flow_from_directory(
    'pokemonData/train', batch_size=1, class_mode='categorical', subset='validation')

trainX = train_data[0][0]
trainY = train_data[0][1]
testX = test_data[0][0]
testY = test_data[0][1]

# Split out trainX, trainY, testX, testY
for i in range(len(train_data)-1):
    trainX = np.concatenate((trainX, train_data[i][0]))
    trainY = np.concatenate((trainY, train_data[i][1]))
for i in range(len(test_data)-1):
    testX = np.concatenate((testX, test_data[i][0]))
    testY = np.concatenate((testY, test_data[i][1]))


testX.shape = (153, 196608)
trainX.shape = (656, 196608)
testY.shape = (2754)
print(testY)
trainY.shape = (11808)
print(trainY)
# Load models
# Convolutional Neural Net (1-block VGG)


# # def define_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu',
#                      kernel_initializer='he_uniform', padding='same', input_shape=(256, 256, 3)))
#     model.add(Conv2D(32, (3, 3), activation='relu',
#                      kernel_initializer='he_uniform', padding='same'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
#     model.add(Dense(18, activation='softmax'))
#     # compile model
#     opt = SGD(lr=0.001, momentum=0.9)
#     model.compile(optimizer=opt, loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model


# model = define_model()
# model.fit_generator(train_data, epochs=16,
#                     validation_data=test_data, validation_steps=8)

# loss = model.evaluate_generator(test_data, steps=24)
# print(loss)

# XGBoost Gradient Boosting

kfold = KFold(n_splits=10, random_state=7)
model = xgb.XGBClassifier()
scoring = 'accuracy'
results = cross_val_score(model, trainX, trainY, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))

# data = trainX
# label = trainY
# dtrain = xgb.DMatrix(data, label=label)

# param = {
#     'eta': 0.3,  # the training step for each iteration
#     'silent': 1,  # logging mode - quiet
#     'objective': 'multi:softprob',  # error evaluation for multiclass training
#     'num_class': 18}  # the number of classes that exist in this datset
# num_round = 20

# # %%
# # makedirs('models')


# # %%
# n_members = 5
# for i in range(n_members):
#     model = fit_model(trainX, trainy)
#     filename = 'models/model_' + str(i+1) + '.h5'
#     model.save(filename)
#     print('>Saved %s' % filename)


# # %%

# model = load_model('models/model_1.h5')
# predictions = model.predict(testX)
# testy_enc = to_categorical(testy)


# # %%
# # load models from file
# def load_all_models(n_models):
#     all_models = list()
#     for i in range(n_models):
#         filename = 'models/model_' + str(i+1) + '.h5'
#         model = load_model(filename)
#         all_models.append(model)
#         print('>loaded %s' % filename)
#     return all_models


# # %%
# # create stacked model input dataset as outputs from the ensemble

# def stacked_dataset(members, inputX):
#     stackX = None
#     for model in members:
#         yhat = model.predict(inputX, verbose=0)
#         # stack predictions into [rows, members, probabilities]
#         if stackX is None:
#             stackX = yhat
#         else:
#             stackX = dstack((stackX, yhat))
#     # flatten predictions

#     stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
#     return stackX


# # %%
# def fit_stacked_model(members, inputX, inputy):

#     stackedX = stacked_dataset(members, inputX)
#     model = MLPClassifier(max_iter=1000)
#     model.fit(stackedX, inputy)
#     return model


# # %%
# def stacked_prediction(members, model, inputX):
#     stackedX = stacked_dataset(members, inputX)
#     yhat = model.predict(stackedX)
#     return yhat


# # %%
# # generate 2d classification dataset
# X, y = make_blobs(n_samples=1100, centers=3, n_features=2,
#                   cluster_std=2, random_state=2)
# # split into train and test
# n_train = 100
# trainX, testX = X[:n_train, :], X[n_train:, :]
# trainy, testy = y[:n_train], y[n_train:]


# # %%
# n_members = 5
# members = load_all_models(n_members)
# print('Loaded %d models' % len(members))


# # %%
# # ev standalone models on test dataset

# for model in members:
#     testy_enc = to_categorical(testy)
#     _, acc = model.evaluate(testX, testy_enc, verbose=0)
#     print('Model Accuracy: %.3f' % acc)

# # fit stacked model using ensemble
# model = fit_stacked_model(members, testX, testy)
# # evaluate model on test set
# yhat = stacked_prediction(members, model, testX)
# acc = accuracy_score(testy, yhat)
# print('Stacked Test Accuracy: %.3f' % acc)
