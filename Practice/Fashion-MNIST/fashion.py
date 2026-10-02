#!/usr/bin/env -S uv run --script

import tensorflow.keras  as keras
import numpy             as np
import matplotlib.pyplot as plt

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


class VerySimpleFashionModel(keras.Sequential):
    def __init__(self):
        # Define the model
        super().__init__([
            Input(shape=(784,)),
            Dense(32, activation='relu',   name='L1'),
            Dense(10, activation='linear', name='L2')
        ])

    def build_model(self, X, y, Xv, yv):
        # Compile the model
        super().compile(optimizer=Adam(0.001), loss=SparseCategoricalCrossentropy(from_logits=True))

        # Train the model
        return super().fit(x = X, y = y, epochs=10, verbose=1, validation_data = (Xv, yv))

    def __str__(self):
        return 'VerySimpleFashionModel'


class SimpleFashionModel(keras.Sequential):
    def __init__(self):
        # Define the model
        super().__init__([
            Input(shape=(784,)),
            Dense(128, activation='relu',  name='L1'),
            Dense(64, activation='relu',   name='L2'),
            Dense(10, activation='linear', name='L3')
        ])

    def build_model(self, X, y, Xv, yv):
        # Compile the model
        super().compile(optimizer=Adam(0.001), loss=SparseCategoricalCrossentropy(from_logits=True))

        # Train the model
        return super().fit(x = X, y = y, epochs=10, verbose=1, validation_data = (Xv, yv))

    def __str__(self):
        return 'SimpleFashionModel'


def print_error_rate(model, X, y, set_name):
    logits = model.predict(X, verbose=0)
    pred = np.argmax(logits, axis=1)
    errs = np.sum(pred != y)
    count = len(y)
    print(f"{str(model)} {set_name} error rate = {errs}/{count} = {100 * errs / count:5.3}%")


def fit_and_run(model_reference, X_train, y_train, X_val, y_val, X_test, y_test):
    m = model_reference()
    history = m.build_model(X_train, y_train, X_val, y_val)
    print_error_rate(m, X_train, y_train, "training")
    print_error_rate(m, X_val, y_val, "validation")
    print_error_rate(m, X_test, y_test, "test")

    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    pltfname = f"{str(m)}_learning_curves.png"
    plt.savefig(pltfname, dpi=150, bbox_inches="tight")
    plt.close()


X_train, X_val, X_test, y_train, y_val, y_test = load_fashion_data()
print('--------------------')
fit_and_run(VerySimpleFashionModel, X_train, y_train, X_val, y_val, X_test, y_test)
print('--------------------')
fit_and_run(SimpleFashionModel, X_train, y_train, X_val, y_val, X_test, y_test)

