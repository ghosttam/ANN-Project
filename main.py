# Description: This program detects / predicts if a person has diabetes (1) or not (0)
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# IMPORT DATA #
data = pd.read_csv('diabetes.csv')
print(data.head(7))

# PREPROCESSING THE DATA #
# show the shape (number of rows and columns)
print("Number of Rows and Columns:")
print(data.shape)
# checking for duplicates and removing them
data.drop_duplicates(inplace=True)
# Show the shape to see if any rows were dropped
print(data.shape)
# Show the number of missing (NAN, NaN, na) data for each column
print(data.isnull().sum())

# convert the data into an array
dataset = data.values
print(dataset)

# Slip the data into an independent / feature data set x and a dependent / target data set y.
# Get all of the rows from the first eight columns of the dataset
X = dataset[:, 0:8]
# Get all of the rows from the last column
y = dataset[:, 8]

# Process the feature data set to contain values between 0 and 1 inclusive, by using the min-max scalar method,
# and print the values.
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
print(X_scale)

# slit the training and testing data randomly into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.3, random_state=4)
print("Training Data 70% :", X_train.shape)
print("Testing Data 30% : ", X_test.shape)

# Building ANN model
# - First layer will have 12 neurons and use the ReLu activation function
# - Second layer will have 15 neurons and use the ReLu activation function
# - Third layer and final layer will use 1 neuron and the sigmoid activation function.
model = Sequential([
    Dense(12, activation='relu', input_shape=(8,)),
    Dense(15, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model and give it the ‘binary_crossentropy’ loss function (Used for binary
# classification) to measure how well the model did on training, and then give it the
# Stochastic Gradient Descent ‘sgd’ optimizer to improve upon the loss. Also I want to measure
# the accuracy of the model so add ‘accuracy’ to the metrics.

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model by using the fit method on the training data, and train it in batch sizes of 57, with 1000 epochs.
# Give the model validation data to see how well the model is performing by splitting the training data into 20%
# validation.

# Batch - Total number of training examples present in a single batch
# Epoch - The number of iterations when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
# Fit - Another word for train
hist = model.fit(X_train, y_train,
                 batch_size=57, epochs=1000, validation_split=0.3)

# visualize the training loss and the validation loss to see if the model is overfitting
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# visualize the training accuracy and the validation accuracy to see if the model is overfitting
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

# Make a prediction & print the actual values
prediction = model.predict(X_test)
prediction = [1 if y >= 0.5 else 0 for y in prediction]
# Threshold
print(prediction)
print(y_test)

# evaluate the model on the training data set
pred = model.predict(X_train)
pred = [1 if y >= 0.5 else 0 for y in pred]  # Threshold
print(classification_report(y_train, pred))
print('Confusion Matrix: \n', confusion_matrix(y_train, pred))
print()
print('Accuracy: ', accuracy_score(y_train, pred))
print()

# evaluate the model on the testing data set
pred = model.predict(X_test)
pred = [1 if y >= 0.5 else 0 for y in pred]  # Threshold
print(classification_report(y_test, pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, pred))
print()
print('Accuracy: ', accuracy_score(y_test, pred))
print()