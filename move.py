import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

np.random.seed(3)

classes = 6

# 561-dimensional vector
X_train = np.loadtxt('X_train.txt')
Y_train = np.loadtxt('y_train.txt')
X_test = np.loadtxt('X_test.txt')
Y_test = np.loadtxt('y_test.txt')

# convert to classes
Y_train = keras.utils.to_categorical(Y_train-1, classes)
Y_test = keras.utils.to_categorical(Y_test-1, classes)

model = Sequential()
model.add(Dense(150, input_dim=561, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(.5, seed=5))
model.add(Dense(25, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size= 50, epochs=1000, validation_data=(X_test, Y_test)) 


# Saving the model weights
model.save_weights('activity.h5')
saved_model = model.to_json()
with open('activity.json','w') as f:
    f.write(saved_model)

# Loading the saved model
from keras.models import model_from_json
# Load trained model
# load json and create model
json_file = open('activity', 'r')
loaded_model_json = json_file.read()
json_file.close()
global loaded_model
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("activity.h5")
print("Loaded model from disk")