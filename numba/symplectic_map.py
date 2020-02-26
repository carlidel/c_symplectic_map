import numpy as np
from tqdm import tqdm
import scipy as sc
import scipy.integrate as integrate

import cpu_symplectic_map as cpu


def make_correlated_noise(n_elements, gamma=0.0):
    return cpu.make_correlated_noise(n_elements, gamma)


class symplectic_map(object):
    def __init__(self, omega_0, omega_1, omega_2, epsilon, x_star, delta, alpha, beta, barrier_radius, x_0, p_0):
        """Init symplectic map object!
        
        Parameters
        ----------
        object : self
            self
        omega_0 : float
            Omega 0 frequency
        omega_1 : float
            Omega 1 frequency
        omega_2 : float
            Omega 2 frequency
        epsilon : float
            Noise coefficient
        x_star : float
            X star Nekhoroshev coefficient
        delta : float
            value to avoid singularities for zero action values
        alpha : float
            alpha exponential Nekhoroshev coefficient
        beta : float
            beta polynomial exponent
        barrier_radius : float
            barrier position (x coordinates!)
        x_0 : ndarray
            1D array of x initial positions
        p_0 : ndarray
            1D array of p initial values
        """
        self.omega_0 = omega_0
        self.omega_1 = omega_1
        self.omega_2 = omega_2
        self.epsilon = epsilon
        self.x_star = x_star
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.barrier_radius = barrier_radius
        self.action_radius = barrier_radius ** 2 * 0.5
        self.x_0 = x_0
        self.p_0 = p_0
        self.N = len(x_0)

        self.iterations = 0

        self.x = x_0
        self.p = p_0
        self.times = np.zeros(len(self.x))

    def reset(self):
        """Reset the engine to initial conditions
        """
        self.iterations = 0
        self.x = self.x_0
        self.p = self.p_0
        self.times = np.zeros(len(self.x))

    def compute_common_noise(self, noise_array):
        """Execute iterations with given noise array, common for all particles.
        
        Parameters
        ----------
        noise_array : ndarray
            noise array to use for computation
        """
        self.iterations += len(noise_array)
        self.x, self.p, self.times = cpu.symplectic_map_common(
            self.x, self.p, self.times, noise_array, self.epsilon, self.alpha, self.beta, self.x_star, self.delta, self.omega_0, self.omega_1, self.omega_2, self.action_radius
        )

    def compute_personal_noise(self, n_iterations, gamma=0.0):
        """Execute iterations with correlated noise with different realization for every single particle.
        
        Parameters
        ----------
        n_iterations : unsigned int
            number of iterations to perform
        gamma : float, optional
            correlation coefficient (between 0 and 1!), by default 0.0
        """
        self.iterations += n_iterations
        self.x, self.p, self.times = cpu.symplectic_map_personal(
            self.x, self.p, self.times, n_iterations, self.epsilon, self.alpha, self.beta, self.x_star, self.delta, self.omega_0, self.omega_1, self.omega_2, self.action_radius, gamma
        )

    def get_data(self):
        """Get data from engine.
        
        Returns
        -------
        tuple(ndarray, ndarray, ndarray)
            tuple with x, p, and number of iterations before loss data
        """
        return self.x, self.p, self.times

    def get_filtered_data(self):
        """Get filtered data from engine.
        
        Returns
        -------
        tuple(ndarray, ndarray, ndarray)
            tuple with x, p, and number of iterations before loss data
        """
        t = self.times
        return (self.x)[t == self.iterations], (self.p)[t == self.iterations], t[t == self.iterations]

    def get_action(self):
        """Get action data from engine
        
        Returns
        -------
        ndarray
            action array data
        """
        return (self.x * self.x + self.p * self.p) * 0.5

    def get_filtered_action(self):
        """Get filtered action data from engine (i.e. no zero values of lost particles)
        
        Returns
        -------
        ndarray
            filtered action array data
        """
        action = self.get_action()
        return action[action > 0]

    def get_times(self):
        """Get times from engine
        
        Returns
        -------
        ndarray
            times array
        """
        return self.times

    def get_filtered_times(self):
        """Get only loss times from engine (i.e. only loss particles)
        
        Returns
        -------
        ndarray
            filtered times array
        """
        times = self.get_times()
        return times[times < self.iterations]

    def get_survival_quota(self):
        """Get time evolution of number of survived particles
        
        Returns
        -------
        ndarray
            time evolution of survived particles
        """
        t = np.array(self.get_times())
        max_t = np.amax(t)
        quota = np.empty(max_t)
        for i in range(max_t):
            quota[i] = np.count_nonzero(t > i)
        return quota

    def get_lost_particles(self):
        """Get time evolution of lost particles
        
        Returns
        -------
        ndarray
            time evolution of number of lost particles
        """
        quota = self.get_survival_quota()
        return self.N - quota

    def current_binning(self, bin_size):
        """Execute current binning and computation
        
        Parameters
        ----------
        bin_size : int
            size of the binning to consider for current computation
        
        Returns
        -------
        tuple(ndarray, ndarray)
            array with corresponding sampling time (middle point), current value computed.
        """
        survival_quota = self.get_survival_quota()
        points = [i for i in range(0, len(survival_quota), bin_size)]
        if survival_quota % bin_size == 0:
            points.append(len(survival_quota) - 1)
        t_middle = [(points[i + 1] + points[i]) *
                    0.5 for i in range(len(points) - 1)]
        currents = [(survival_quota[points[i]] - survival_quota[points[i+1]]
                     ) / bin_size for i in range(len(points) - 1)]
        return np.array(t_middle), np.array(currents)
