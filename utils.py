import numpy as np
import os
import pathlib


def _data_folder_path() -> os.PathLike:
    return pathlib.Path('.', 'data')


def _data_file_path(filename: str) -> os.PathLike:
    return pathlib.Path(_data_folder_path(), filename)


def _ensure_data_dir_exists():
    _data_folder_path().mkdir(parents=True, exist_ok=True)


def load_or_generate_and_then_save(filename : os.PathLike, generate_fn : callable) -> np.ndarray:
    '''Checks if filepath exists in the data folder. If it exists, the data is loaded from the path. If not, the data
    is generated from the generation function that is passed as an argument. The data is then saved to the given 
    filename in the data folder, after which the data is returned.
    '''
    _ensure_data_dir_exists()

    path = _data_file_path(filename)
    
    if path.is_file():
        with open(path, 'rb') as f:
            data = np.load(_data_file_path(filename))['arr_0']
    else:
        data = generate_fn()
        with open(path, 'wb') as f:
            np.savez(f, data)

    return data