import numpy as np
import tensorflow as tf

# Get the MNIST data and preprocess it
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
num_classes = 10

# Scale images to the [0, 1] range
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
print("x_train shape:", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

# convert class vectors to binary class matrices (one-hot coding?)
y_hc_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_hc_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Generate the TF/Keras model (We are going to replicate this model as an SNN in every framework)
kerasModel = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28), name="input"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(units=128, activation='relu', use_bias=False, name="hidden"),
        tf.keras.layers.Dense(units=num_classes, activation='softmax', use_bias=False, name="output")
    ]
)

kerasModel.summary()

# Train the keras model
kerasModel.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                   loss=tf.keras.losses.CategoricalCrossentropy(),
                   metrics=['accuracy'])

history = kerasModel.fit(x=X_train, y=y_hc_train, batch_size=256, epochs=150, verbose=0)
accuracy = kerasModel.evaluate(x=X_test, y=y_hc_test)[1]
print(accuracy)

# Save the model weights to be reused in other neurons
# kerasModel.save('model_weights.h5')
