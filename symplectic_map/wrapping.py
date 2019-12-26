import numpy as np
from tqdm import tqdm
import scipy as sc
import scipy.integrate as integrate

from c_symplectic_map import c_symplectic_map

class symplectic_map(object):
    def __init__(self, omega_0, omega_1, omega_2, epsilon, x_star, delta, alpha, beta, barrier_radius, x_0, p_0):
        self.omega_0 = omega_0
        self.omega_1 = omega_1
        self.omega_2 = omega_2
        self.epsilon = epsilon
        self.x_star = x_star
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.barrier_radius = barrier_radius
        self.x_0 = x_0
        self.p_0 = p_0
        self.N = len(x_0)

        self.engine = c_symplectic_map(omega_0, omega_1, omega_2, epsilon, x_star, delta, alpha, beta, barrier_radius, x_0, p_0)

    def reset(self):
        self.engine.reset()

    def compute_common_noise(self, noise_array):
        self.engine.compute(len(noise_array), 1, noise_array)
    
    def compute_personal_noise(self, n_iterations, gamma=0.0):
        self.engine.compute(n_iterations, 1, gamma)

    def get_data(self):
        return np.array(self.engine.x()), np.array(self.engine.p()), np.array(self.engine.t())

    def get_action(self):
        x = np.array(self.engine.x())
        p = np.array(self.engine.p())
        return (x * x + p * p) * 0.5

    def get_times(self):
        return np.array(self.engine.t())

    def get_survival_quota(self):
        t = np.array(self.get_times())
        max_t = np.amax(t)
        quota = np.empty(max_t)
        for i in range(max_t):
            quota[i] = np.count_nonzero(t > i)
        return quota

    def get_lost_particles(self):
        quota = self.get_survival_quota()
        return self.N - quota

    def current_binning(self, bin_size):
        survival_quota = self.get_survival_quota()
        points = [i for i in range(0, len(survival_quota), bin_size)]
        if survival_quota % bin_size == 0:
            points.append(len(survival_quota) - 1)
        t_middle = [(points[i + 1] + points[i]) * 0.5 for i in range(len(points) - 1)]
        currents = [(survival_quota[points[i]] - survival_quota[points[i+1]]) / bin_size for i in range(len(points) - 1)]
        return np.array(t_middle), np.array(currents)

    
