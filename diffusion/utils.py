import numpy as np
import os

def load_or_generate_and_then_save(filepath : os.PathLike, generate_fn : callable) -> np.ndarray:
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            data = np.load(filepath)['arr_0']
    else:
        data = generate_fn()
        with open(filepath, 'wb') as f:
            np.savez(f, data)
    return data
