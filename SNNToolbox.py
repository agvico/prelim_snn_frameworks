import numpy as np
import tensorflow as tf
from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser
import os
from shutil import copyfile

pathwd = "./SNNToolbox"
datadir = pathwd + "/mnist"
os.mkdir(pathwd)
os.mkdir(datadir)

presentation_time = 350

# Get the MNIST data and preprocess it
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
num_classes = 10

# convert class vectors to binary class matrices (one-hot coding?)
y_hc_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_hc_test = tf.keras.utils.to_categorical(y_test, num_classes)

np.savez(datadir + "/x_test", X_test)
np.savez(datadir + "/y_test", y_hc_test)


copyfile('model_weights.h5', pathwd + "/model_weights.h5")

# SNN TOOLBOX CONFIGURATION #
#############################

# Create a config file with experimental setup for SNN Toolbox.
configparser = import_configparser()
config = configparser.ConfigParser()

config['paths'] = {
    'path_wd': pathwd,              # Path to model.
    'dataset_path': datadir,        # Path to dataset.
    'filename_ann': "model_weights"      # Name of input model.
}

config['tools'] = {
    'evaluate_ann': True,            # Test ANN on dataset before conversion.
    'normalize': False               # Normalize weights for full dynamic range.
}

config['simulation'] = {
    'simulator': 'INI',             # Chooses execution backend of SNN toolbox.
    'duration': presentation_time,                 # Number of time steps to run each sample.
    'num_to_test': X_test.shape[0],             # How many test samples to run.
    'batch_size': 50,               # Batch size for simulation.
    'keras_backend': 'tensorflow'   # Which keras backend to use.
}

config['output'] = {
    'verbose': 0        # Non-verbose
}

# Store config file.
config_filepath = os.path.join(pathwd, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

# RUN SNN TOOLBOX #
###################

main(config_filepath)
