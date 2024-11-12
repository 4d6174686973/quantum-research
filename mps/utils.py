import numpy as np
import pandas as pd


def sample_info(samples_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract the sample information from a dictionary with form {bitstring: count}."""
    values = np.array([np.array([int(i) for i in bitstring]) for bitstring in samples_dict.keys()])
    probabilities = np.array(list(samples_dict.values())) / sum(samples_dict.values())
    return values, probabilities


def array_to_str(binary_array: np.ndarray) -> list[str]:
    """ Convert binary array of form [[1,1,1],[1,0,1],...] to bitstring array
    of form ["111","101",...] needed for counting """
    str_dataset = []
    for i in range(binary_array.shape[0]):
        str_dataset.append("".join(str(int(j)) for j in binary_array[i]))
    return np.array(str_dataset)


def real_to_binary(data: np.ndarray, bits_per_feature: int):
    '''Conversion of real-valued data set into binary features. Every real valued number is 
    converted into a n-bit binary number.
    
    Parameters
    -----------
    data: DataFrame
        The real-valued data set of shape (n_samples, n_features).
    
    Returns
    --------
    data_binary: DataFrame
        The binary data set fo shape (n_samples, bits_per_feature * n_features).'''
    data_binary = np.array([[0] * (bits_per_feature * data.shape[1]) for _ in range(data.shape[0])])
    x_min, x_max = np.min(data, axis=0), np.max(data, axis=0)
    for n in range(data.shape[1]):
        for l in range(data.shape[0]):
            x_integer = int((2**bits_per_feature - 1) * (data[l, n] - x_min[n]) / (x_max[n] - x_min[n]))
            binary_string = bin(x_integer)[2:].zfill(bits_per_feature)
            for bit_index, bit in enumerate(binary_string):
                data_binary[l, n * bits_per_feature + bit_index] = int(bit)
    return data_binary, [x_min, x_max]


def binary_to_real(X_binary, X_min, X_max, bits_per_feature):
    """
    Converts a set of binary features back into real-valued data.

    Parameters
    -----------
    X_binary: DataFrame
        The binary data set of shape (n_samples, bits_per_feature * n_features).
    X_min: float or array-like
        The minimum value for each feature (output of from_real_to_binary).
    X_max: float or array-like
        The maximum value for each feature (output of from_real_to_binary).
    
    Returns
    --------
    X_real: DataFrame
        The real-valued data set of shape (n_samples, n_features).
    """
    N_samples = len(X_binary)
    if isinstance(X_min, float):
        N_variables = 1
    else:
        N_variables = len(X_min)
    X_real = [[0] * N_variables for _ in range(N_samples)]

    for n in range(N_variables):
        for l in range(N_samples):
            X_integer = sum(X_binary[l][n * bits_per_feature + m] * (2 ** (bits_per_feature - 1 - m)) for m in range(bits_per_feature))
            X_real[l][n] = X_min[n] + (X_integer * (X_max[n] - X_min[n]) / ((2 ** bits_per_feature) -1))
    
    X_real = np.array(X_real)

    return X_real
