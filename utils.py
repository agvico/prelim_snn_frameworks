from typing import Dict

import numpy as np
import h5py


def load_weights(path: str) -> Dict[str, np.array]:
    # Load the weights of the pre-trained ANN
    weights = {}
    f = h5py.File(path, 'r')
    weights['inpToHidden'] = np.array(f['model_weights']['hidden']['hidden']['kernel:0'])
    weights['hiddenToOut'] = np.array(f['model_weights']['output']['output']['kernel:0'])
    assert weights['inpToHidden'].shape == (784, 128)
    assert weights['hiddenToOut'].shape == (128, 10)
    return weights
