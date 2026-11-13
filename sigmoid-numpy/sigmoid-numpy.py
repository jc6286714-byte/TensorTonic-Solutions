import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    array = np.asarray(x, dtype=float)
    # Ensuring numerical stability
    output = np.where(
        array >= 0, 
        1.0 / (1.0 + np.exp(-np.abs(array))),
        np.exp(-np.abs(array)) / (1.0 + np.exp(-np.abs(array))), 
    )
    if np.ndim(x) == 0:
        return float(output)
    return output