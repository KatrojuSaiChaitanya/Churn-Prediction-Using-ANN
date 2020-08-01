# Artificial neural network

# importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Reading the csv file

df = pd.read_csv('C:/Users/Chaitanya/Desktop/Pylab/Churn_Modelling.csv')

# printing the dataset

print(df)

# Assigning dependent and independent variables

X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values

labelencoder_X_2 = LabelEncoder()

# apply label encoding for the gender column

X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Assigning dummy values to the categorical data
# since we have Geography and Gender as categorical data in our data set we use either One hot encoding or get_dummies

transformer = ColumnTransformer(transformers=[("dummy",OneHotEncoder(categories='auto'),[1])],remainder='passthrough')
X = transformer.fit_transform(X.tolist())

X = X[:, 1:]
# Now our data is ready to be split into training and testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print('Training data')
print('independent data :\n', X_train)
print('dependent data :\n', y_train)
print('Testing data')
print('independent data :\n', X_test)
print('dependent data :\n', y_test)

'''
Now since in Ann the hidden layer multiplies the weights with input features, if the value is large enough, then it
might take longer time for execution, inorder to reduce this we have to use standard scaler to normalize the data'''

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# printing the final dependent and independent data of training data set after normalization

print(X_train)
print(X_test)

# We are now ready to build a ANN model

# importing important libraries

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout

# create an empty neural network

s = Sequential()


# add the first layer
# kernal initializer plays a major role which is used in initializing the weights
# units are the number of neurons to be created in the hidden layer
# input_dim represents the number of features takes as inputs
# as we know that it is the activation function which stimulates the neuron we use activation
# we should basically use relu activation in the hidden layers and sigmoid at the output layers

s.add(Dense(units = 6, kernel_initializer= 'he_uniform',activation='relu',input_dim = 11))

# adding second hidden layer

s.add(Dense(units=6, kernel_initializer='he_uniform',activation='relu'))

# adding the output layer

s.add(Dense(activation="sigmoid", units=1, kernel_initializer="glorot_uniform"))

# Now the Artificial neural network has been built with a total of 3 layers 2-hidden and 1-output

# In order to reduce the loss function we use optimizers like gradient descent, stochastic gradient descent, adagrad...

# The best is adam optimizer
# when there is only 2 categorical labels in the output we use binary_crossentropy or else we use
# categorical_crossentropy

s.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

# ANN model is built and compiled, now we have to fit the model to the training data

model_history = s.fit(X_train, y_train,validation_split=0.33, batch_size = 10, epochs = 5)

# list all data in history

print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = s.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)

"""


# Initialising the ANN

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'he_uniform',activation='relu',input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'he_uniform',activation='relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'glorot_uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, nb_epoch = 100)

# list all data in history

print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)

"""
