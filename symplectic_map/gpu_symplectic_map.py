import numpy as np
from numba import njit, cuda
from numba.cuda import random
import math
from numba import float64

@cuda.jit
def symplectic_map_common(x, px, step_values, noise_array, epsilon, alpha, beta, x_star, delta, omega_0, omega_1, omega_2, action_radius):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    action = cuda.shared.array(shape=(512), dtype=float64)
    rot_angle = cuda.shared.array(shape=(512), dtype=float64)
    temp1 = cuda.shared.array(shape=(512), dtype=float64)
    temp2 = cuda.shared.array(shape=(512), dtype=float64)

    if j < x.shape[0]:
        for k in range(noise_array.shape[0]):
            action[i] = (x[j] * x[j] + px[j] * px[j]) * 0.5
            rot_angle[i] = omega_0 + (omega_1 + action[i]) + (0.5 * omega_2 * action[i] * action[i])

            if (x[j] == 0.0 and px[j] == 0.0) or action[i] >= action_radius:
                x[j] = 0.0
                px[j] = 0.0
                break

            temp1[i] = x[j]
            temp2[i] = (
                px[j] + epsilon * noise_array[k] * (x[j] ** beta) 
                * math.exp(-((x_star / (delta + abs(x[j]))) ** alpha))
            )
            x[j] = math.cos(rot_angle[i]) * temp1[i] + \
                math.sin(rot_angle[i]) * temp2[i]
            px[j] = -math.sin(rot_angle[i]) * temp1[i] + \
                math.cos(rot_angle[i]) * temp2[i]

            step_values[j] += 1


@cuda.jit
def symplectic_map_personal(x, px, step_values, n_iterations, epsilon, alpha, beta, x_star, delta, omega_0, omega_1, omega_2, action_radius, rng_states, gamma):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    action = cuda.shared.array(shape=(512), dtype=float64)
    rot_angle = cuda.shared.array(shape=(512), dtype=float64)
    temp1 = cuda.shared.array(shape=(512), dtype=float64)
    temp2 = cuda.shared.array(shape=(512), dtype=float64)
    noise = cuda.shared.array(shape=(512), dtype=float64)

    noise[i] = random.xoroshiro128p_normal_float64(rng_states, j)

    if j < x.shape[0]:
        for k in range(n_iterations):
            action[i] = (x[j] * x[j] + px[j] * px[j]) * 0.5
            rot_angle[i] = omega_0 + (omega_1 + action[i]) + \
                (0.5 * omega_2 * action[i] * action[i])

            if (x[j] == 0.0 and px[j] == 0.0) or action[i] >= action_radius:
                x[j] = 0.0
                px[j] = 0.0
                break
            
            temp1[i] = x[j]
            temp2[i] = (
                px[j] + epsilon * noise[i] * (x[j] ** beta)
                * math.exp(-((x_star / (delta + abs(x[j]))) ** alpha))
            )
            x[j] = math.cos(rot_angle[i]) * temp1[i] + \
                math.sin(rot_angle[i]) * temp2[i]
            px[j] = -math.sin(rot_angle[i]) * temp1[i] + \
                math.cos(rot_angle[i]) * temp2[i]

            step_values[j] += 1

            noise[i] = random.xoroshiro128p_normal_float64(rng_states, j) + gamma * noise[i]
