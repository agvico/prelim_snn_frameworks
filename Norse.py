import numpy as np
import tensorflow as tf
import torch
from utils import load_weights
import norse.torch as norse
import time

# load mnist data (do not normalize input)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# Flatten the images:
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# BindsNet works with torch.Tensor, so convert data to Tensor:
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)

# Load the weights of the pre-trained ANN:
weights = load_weights('model_weights.h5')

# Define the SNN layers parameters:
n_input = 784
n_hidden = 128
n_out = 10
presentation_time = 350 #ms
dt = 1.0


# For Poisson encoder, inputs are expect to be in the [0,1] range
X_test = X_test / 255.0 # torch.Tensor(np.tile(X_test, (presentation_time, 1, 1)))

# Define the network
model = norse.SequentialState(
    norse.PoissonEncoder(seq_length=presentation_time, f_max=255),  # Poisson input encoding
    norse.Lift(torch.nn.Linear(n_input, n_hidden, bias=False)),     # Lift is for applying the layer over time.
    norse.LIF(),
    norse.Lift(torch.nn.Linear(n_hidden, n_out, bias=False)),
    norse.LIF()
)

t_start = time.time()

# set the network weights
with torch.no_grad():
    model[1].lifted_module.weight = torch.nn.Parameter(torch.Tensor(np.transpose(weights['inpToHidden'])))
    model[3].lifted_module.weight = torch.nn.Parameter(torch.Tensor(np.transpose(weights['hiddenToOut'])))

    # run the simulation network:
    out, _ = model(X_test)
    y_preds = torch.argmax(torch.sum(out, dim=0), dim=1).detach().numpy()

acc = np.sum(y_preds == y_test) / len(y_test)
print("Accuracy: ", acc)
print("Exec time: ", (time.time() - t_start), "ms")
