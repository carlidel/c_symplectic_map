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
        corr_noise = np.empty_like(noise)
        corr_noise[0] = noise[0]
        corr_noise[1:] = np.array([noise[i-1] + gamma * noise[i] for i in range(1, n_elements)])
        return corr_noise
    return noise
