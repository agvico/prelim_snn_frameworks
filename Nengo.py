import numpy as np
import tensorflow as tf
from nengo.solvers import NoSolver
from utils import load_weights
from tqdm import tqdm
import nengo
import nengo_dl
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
presentation_time = 350   # Time to present an example to the network

# Nengo needs instances with 3-D shape: (num_instances, n_timesteps, n_features), so we need to add the time dimension
# In this case, we are going to present the image to the network for the presentation time, so we tile it for that
# amount of time:
X_test = np.tile(X_test[:, None, :], reps=(1, presentation_time, 1))

# Define the network
with nengo.Network(seed=0) as net:
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])  # Max firing rate is 100Hz ??
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None
    net.config[nengo.Connection].solver = NoSolver()

    # Define the layers
    inp = nengo.Node(output=np.ones(n_input))
    poissonInput = nengo.Ensemble(n_neurons=n_input, dimensions=1,
                                  neuron_type=nengo.PoissonSpiking(nengo.RectifiedLinear(), amplitude=1))
    hidden = nengo.Ensemble(n_neurons=n_hidden, dimensions=1, neuron_type=nengo.LIF())
    out = nengo.Ensemble(n_neurons=n_out, dimensions=1, neuron_type=nengo.LIF())

    # Define the connections
    nengo.Connection(inp, poissonInput.neurons)
    inpToHidden = nengo.Connection(pre=poissonInput.neurons, post=hidden.neurons,
                                   transform=np.transpose(weights['inpToHidden']))
    hiddenToOut = nengo.Connection(pre=hidden.neurons, post=out.neurons,
                                   transform=np.transpose(weights['hiddenToOut']))
    # Add monitors (Probes)
    out_probe = nengo.Probe(out.neurons, synapse=0.1)

# Run
t_start = time.time()
with nengo_dl.Simulator(net) as sim:
    y_preds = np.zeros(X_test.shape[0])
    for i in tqdm(range(X_test.shape[0])):
        sim.run_steps(presentation_time, data={inp: X_test[i][None, :, :]}, progress_bar=False)
        y_preds[i] = np.argmax(np.sum(sim.data[out_probe] > 0, axis=0))
        sim.reset()

    acc = np.sum(y_preds == y_test) / len(y_test)
    print("Accuracy: ", acc)

print("Exec. time: ", (time.time() - t_start), "ms")