import numpy as np
from .wrapping import symplectic_map

def make_correlated_noise(n_elements, gamma=0.0):
    """Make a correlated random noise
    
    Parameters
    ----------
    n_elements : unsigned int
        number of elements desired
    gamma : float, optional
        correlation coefficient, by default 0.0
    
    Returns
    -------
    ndarray
        noise array
    """    
    noise = np.random.normal(size=n_elements)
    if gamma != 0.0:
        corr_noise = []
        corr_noise.append(noise[0])
        for i in range(1, n_elements):
            corr_noise.append(corr_noise[-1] * gamma + noise[i])
        return np.asarray(corr_noise)
    return noise
