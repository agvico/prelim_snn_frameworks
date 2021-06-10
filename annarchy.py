import numpy as np
import tensorflow as tf
from utils import load_weights
from tqdm import tqdm
import time
import ANNarchy
from ANNarchy import IF_curr_exp, Synapse, Projection, Population, PoissonPopulation, Monitor, compile, simulate


# load mnist data (do not normalize input)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# Flatten the images:
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# Load the weights of the pre-trained ANN:
weights = load_weights('model_weights.h5')

# Define the SNN in brian:
n_input = 784
n_hidden = 128
n_out = 10
presentation_time = 350.0  # Time to present an example to the network
resting_time = 100.0      # Resting time of the network, where no input is shown in order to restart voltages.
expected_runtime = X_test.shape[0] * (presentation_time + resting_time)


# Network creation (layers)
lif_neuron = IF_curr_exp(v_rest=0.0, v_reset=0.0, v_thresh=5.0)
input = PoissonPopulation(geometry=n_input, rates=100.0, name="input")
hidden = Population(geometry=n_hidden, neuron=lif_neuron, name="hidden")
out = Population(geometry=n_out, neuron=IF_curr_exp(v_rest=0.0, v_reset=0.0, v_thresh=2.0), name="output")

# Connect layers
DefaultSynapse = Synapse(
    parameters = "w=0.0",
    equations = "",
    pre_spike = """
        g_target += w
    """
)
inpToHidden = Projection(pre=input, post=hidden, target='exc', name='inpToHidden', synapse=DefaultSynapse)
hiddenToOut = Projection(pre=hidden, post=out, target='exc', name='hiddenToOut', synapse=DefaultSynapse)

# Apply the connection and load the ANN weights
inpToHidden.connect_from_matrix(weights=np.transpose(weights['inpToHidden']))
hiddenToOut.connect_from_matrix(weights=np.transpose(weights['hiddenToOut']))

# Define monitor for output spikes
m = Monitor(out, ['v', 'spike'])

## COMPILE AND RUN THE SIMULATION for each test instance
compile(clean=True)
t_start = time.time()
y_preds = np.zeros(X_test.shape[0])
for i in tqdm(range(X_test.shape[0])):
    input.rates = X_test[i]
    simulate(presentation_time)

    # Retrieve the number of spikes per neuron. The maximum is the final classification value
    spikes = m.get('spike')
    spike_count = [len(i) for i in spikes.values()]
    y_preds[i] = np.argmax(spike_count)

    # Reset the network to predict the next instance
    ANNarchy.reset()

# Compute Accuracy
acc = np.sum(y_preds == y_test) / len(y_test)
print("Accuracy: ", acc)
print("Exec. time: ", (time.time() - t_start), "ms.")
