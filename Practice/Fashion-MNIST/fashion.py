#!/usr/bin/env -S uv run --script

import tensorflow.keras as keras

from keras            import Sequential
from keras.optimizers import Adam
from keras.layers     import Dense, Input
from keras.losses     import SparseCategoricalCrossentropy

(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

print(X_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)

#print(X_test.shape)  # (60000, 28, 28)
#print(y_test.shape)  # (60000,)

X_train = X_train / 255.0
X_train = X_train.reshape(-1, 784)

X_test = X_test / 255.0
X_test = X_test.reshape(-1, 784)

# Define the model
model = Sequential([
            Input(shape=(784,)),
            Dense(32, activation='relu'  , name='L1'),
            Dense(10, activation='linear', name='L2')
        ])

# Compile the model
model.compile(
    optimizer=Adam(0.001),
    loss=SparseCategoricalCrossentropy(from_logits=True)
)

# Train the model
history = model.fit(
    x = X_train,
    y = y_train,
    epochs=10,
    verbose=1
)

#print(model.get_layer('L1').get_weights())
#print(model.get_layer('L2').get_weights())
#print(history.history)



