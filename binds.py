import numpy as np
import tensorflow as tf
import torch
from bindsnet.network.monitors import Monitor
from utils import load_weights
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.encoding import poisson_loader
from tqdm import tqdm


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

################################
# Define the SNN of BindsNet
################################
network = Network(learning=False)

#  input layer (will be a Poisson generator later).
input = Input(n=n_input, shape=(n_input,1))
network.add_layer(input, name='input')

# Hidden LIF layer (we let BindsNet defaults)
hidden = LIFNodes(n=n_hidden, rest=0, thresh=5, reset=0)
network.add_layer(hidden, name='hidden')

# Output layer (linear read-out, we added )
out = LIFNodes(n=n_out, rest=0, thresh=2, reset=0)
network.add_layer(out, name='output')

# Add connections with the pretrained weights:
inpToHid = Connection(source=input, target=hidden, w=torch.Tensor(weights['inpToHidden']))
hidToOut = Connection(source=hidden, target=out, w=torch.Tensor(weights['hiddenToOut']))
network.add_connection(inpToHid, source='input', target='hidden')
network.add_connection(hidToOut, source='hidden', target='output')

# Adds the output spikes monitor (for determine the classification)
# Create a monitor.
out_monitor = Monitor(
    obj=out,
    state_vars=("s", "v"),   # Record spikes and voltages.
    time=presentation_time,  # Length of simulation (if known ahead of time).
)
network.add_monitor(out_monitor, name="output_monitor")

####################################
#   RUN THE SIMULATION
####################################
# Define the input data as poisson spikes and inject into the input layer.
input_data = poisson_loader(data=X_test, time=presentation_time, dt=dt)

# Now, for each sample: present it to the network, get its prediction and reset its state
y_preds = []

for i in tqdm(input_data):
    inputs = {"input": i}
    network.run(inputs=inputs, time=presentation_time)

    # Collect classification output as the output neuron
    spikes = out_monitor.get('s')
    spike_count = torch.sum(spikes, dim=0)
    y_pred = torch.argmax(spike_count)
    y_preds.append(y_pred)

    # Reset the network for the next instance.
    network.reset_state_variables()

# Compute Accuracy
y_preds = np.array(y_preds)
acc = np.sum(y_preds == y_test) / len(y_test)

print("Accuracy: ", acc)
