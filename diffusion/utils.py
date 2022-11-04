import numpy as np
import os

def load_or_generate_and_then_save(filepath : os.PathLike, generate_fn : callable) -> np.ndarray:
    '''Checks if filepath exists. If it exists, the data is loaded from the path. If not, the data
    is generated from the generation function that is passed as an argument. The data is then saved
    to the given filepath, after which the data is returned.
    '''
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            data = np.load(filepath)['arr_0']
    else:
        data = generate_fn()
        with open(filepath, 'wb') as f:
            np.savez(f, data)
    return data
