import numpy as np
import tensorflow as tf
from brian2 import *
from utils import load_weights
from tqdm import tqdm
import time

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
presentation_time = 350*ms  # Time to present an example to the network
resting_time = 100*ms       # Resting time of the network, where no input is shown in order to restart voltages.
expected_runtime = X_test.shape[0] * (presentation_time + resting_time)

# Start the BRIAN SCOPE
print('building net...')
# Define the input as a Poisson spike generation using the images numbers as the firing rate:
input = PoissonGroup(N=n_input, rates=X_test[0]*Hz)

# LIF neuron equation:
tau = 10*ms
lif_eqs = '''
dv/dt = -v/ tau : volt (unless refractory)
'''

# Define the 'layers' of the SNN
hidden = NeuronGroup(n_hidden, lif_eqs, threshold='v > 5*mV', reset='v = 0*mV',
                            refractory=5*ms, method='exact')
# initialise neuron volatage
hidden.v = 0*mV

# Output is a linear read-out, i.e, it fires whenever a spike arrives
out = NeuronGroup(n_out, lif_eqs, threshold='v > 2*mV', reset='v = 0*mV',
                            refractory=5*ms, method='exact')
out.v = 0*mV

# Create the synapses (No learning method). Note how the neuron voltage is added here (on_pre)
inpToHid = Synapses(source=input, target=hidden, model='''w: volt''', on_pre='v += w')
hidToOut = Synapses(source=hidden, target=out, model='''w: volt''', on_pre='v += w')

# Connect them all. Fully-connected layers
inpToHid.connect()
hidToOut.connect()

# Set the connection weights from the pre-trained ANN:
inpToHid.w = weights['inpToHidden'].flatten() * mV
hidToOut.w = weights['hiddenToOut'].flatten() * mV



print('running...')
y_preds = np.zeros(X_test.shape[0])  # This array stores the predictions of the SNN for each test sample.

start_time = time.time()
##################
#  RUN THE SIMULATION WITH THE TEST SET OF MNIST
##################
for i in tqdm(range(X_test.shape[0])):  # For example in test:
    # Set monitors:
    out_spike_monitor = SpikeMonitor(out)
    hid_mon = SpikeMonitor(hidden)
    inp_mon = SpikeMonitor(input)

    # Set the new example as the rates for the Poisson input
    input.rates = X_test[i] * Hz

    # Run the network for the presentation time: Present an input for a given period of time
    run(presentation_time)

    # Retrieve the spike count of the last layer. The maximum value is the classification output value:
    spikes = out_spike_monitor.count
    y_preds[i] = np.argmax(spikes)

    # reset the network. No input for the resting time
    # Reset monitors
    input.rates = 0 * Hz
    out_spike_monitor = SpikeMonitor(out)
    hid_mon = SpikeMonitor(hidden)
    inp_mon = SpikeMonitor(input)
    run(resting_time)

# Get the classification accuracy
acc = np.sum(y_preds == y_test) / len(y_test)
end_time = time.time()
runtime = end_time - start_time

print("Accuracy: ", acc)
print("Execution time: ", runtime)
