#!/usr/bin/env -S uv run --script

import tensorflow.keras as keras
import numpy            as np

from keras            import Sequential
from keras.optimizers import Adam
from keras.layers     import Dense, Input
from keras.losses     import SparseCategoricalCrossentropy
from fractions        import Fraction

from sklearn.model_selection import train_test_split

# Set a seed before defining the model for predictable results
keras.utils.set_random_seed(8462)

def load_fashion_data():
    # Load the MNIST Fashion dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,       # an integer like 10000 or a fraction like 0.2
        random_state=42,     # makes the split reproducible
        stratify=y_train     # keeps class proportions equal in both sets
    )

    # Scale and reshape the training and test vectors
    X_train = X_train / 255.0
    X_train = X_train.reshape(-1, 784)

    X_test = X_test / 255.0
    X_test = X_test.reshape(-1, 784)

    X_val = X_val / 255.0
    X_val = X_val.reshape(-1, 784)

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_model_error_rate(model, X, y):
    logits = model.predict(X)
    pred = np.argmax(logits, axis=1)
    errs = np.sum(pred != y)
    count = len(y)
    return Fraction(errs, count)


def define_and_train_model(X, y):
    # Define the model
    model = Sequential([
                Input(shape=(784,)),
                Dense(32, activation='relu'  , name='L1'),
                Dense(10, activation='linear', name='L2')
            ])

    # Compile the model
    model.compile(optimizer=Adam(0.001), loss=SparseCategoricalCrossentropy(from_logits=True))

    # Train the model
    model.fit(x = X, y = y, epochs=10, verbose=1)

    return model


X_train, X_val, X_test, y_train, y_val, y_test = load_fashion_data()

model = define_and_train_model(X_train, y_train)

train_err = get_model_error_rate(model, X_train, y_train)
print(f"training error rate = {str(train_err)} = {100 * float(train_err):0.3}%")

test_err = get_model_error_rate(model, X_test, y_test)
print(f"test error rate = {str(test_err)} = {100 * float(test_err):0.3}%")

val_err = get_model_error_rate(model, X_val, y_val)
print(f"validation error rate = {str(val_err)} = {100 * float(val_err):0.3}%")

