import tensorflow.keras as keras
import pathlib
import datetime

path = pathlib.Path('resources/mnist.npz').absolute()
(x_train, y_train), (x_valid, y_valid) = keras.datasets.mnist.load_data(path)
# data is of type 'uint8' (values from 0 to 255)

num_categories = 10 # 10 possible digits, 0-9

# reshape image data from (..., 28, 28) to (..., 784)
x_train = x_train.reshape((-1, 784))
x_valid = x_valid.reshape((-1, 784))

# normalize the image data to be in [0.0, 1.0]
x_train = x_train / 255
x_valid = x_valid / 255

# turn category integers (0-9) into one-hot vectors (ex: 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y_train = keras.utils.to_categorical(y_train, num_categories)
y_valid = keras.utils.to_categorical(y_valid, num_categories)

# create model
model = keras.models.Sequential()
# first layer: 512 nodes with relu activation
model.add(keras.layers.Dense(units=512, activation='relu', input_shape=(784,)))
# second layer: 512 nodes with relu activation
model.add(keras.layers.Dense(units=512, activation='relu'))
# output layer: 10 nodes (one for each digit) with softmax activation
model.add(keras.layers.Dense(units=10, activation='softmax'))
print(model.summary())

# compile model to configure for training - specify loss function and what metrics to track
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# train the model - can specify batch size and number of epochs
# batch_size defaults to None - 32 samples per gradient update
# fitting returns a History object which contains information about metrics
history = model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1, validation_data=(x_valid, y_valid))

# export model to use later on
model.save('models/mnist_model_' + str(datetime.datetime.now())[:19].replace(':', '-').replace(' ', '_'))
