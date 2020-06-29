import numpy as np
from numba import njit, cuda
from numba.cuda import random
import math
from numba import float64, int32

@cuda.jit
def symplectic_map_common(x, px, step_values, noise_array, epsilon, alpha, beta, x_star, delta, omega_0, omega_1, omega_2, action_radius):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    action = cuda.shared.array(shape=(512), dtype=float64)
    rot_angle = cuda.shared.array(shape=(512), dtype=float64)
    temp1 = cuda.shared.array(shape=(512), dtype=float64)
    temp2 = cuda.shared.array(shape=(512), dtype=float64)
    l_x = cuda.shared.array(shape=(512), dtype=float64)
    l_px = cuda.shared.array(shape=(512), dtype=float64)
    l_step = cuda.shared.array(shape=(512), dtype=int32)
    
    if j < x.shape[0]:
        l_x[i] = x[j]
        l_px[i] = px[j]
        l_step[i] = step_values[j]
        for k in range(noise_array.shape[0]):
            action[i] = (l_x[i] * l_x[i] + l_px[i] * l_px[i]) * 0.5
            rot_angle[i] = omega_0 + (omega_1 + action[i]) + (0.5 * omega_2 * action[i] * action[i])

            if (l_x[i] == 0.0 and px[i] == 0.0) or action[i] >= action_radius:
                l_x[i] = 0.0
                l_px[i] = 0.0
                break

            temp1[i] = l_x[i]
            temp2[i] = (
                l_px[i] + epsilon * noise_array[k] * (l_x[i] ** beta)
                * math.exp(-((x_star / (delta + abs(l_x[i]))) ** alpha))
            )
            l_x[i] = math.cos(rot_angle[i]) * temp1[i] + \
                math.sin(rot_angle[i]) * temp2[i]
            l_px[i] = -math.sin(rot_angle[i]) * temp1[i] + \
                math.cos(rot_angle[i]) * temp2[i]

            l_step[i] += 1
        x[j] = l_x[i]
        px[j] = l_px[i]
        step_values[j] = l_step[i]


@cuda.jit
def symplectic_map_personal(x, px, step_values, n_iterations, epsilon, alpha, beta, x_star, delta, omega_0, omega_1, omega_2, action_radius, rng_states, gamma):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    action = cuda.shared.array(shape=(512), dtype=float64)
    rot_angle = cuda.shared.array(shape=(512), dtype=float64)
    temp1 = cuda.shared.array(shape=(512), dtype=float64)
    temp2 = cuda.shared.array(shape=(512), dtype=float64)
    noise = cuda.shared.array(shape=(512), dtype=float64)
    l_x = cuda.shared.array(shape=(512), dtype=float64)
    l_px = cuda.shared.array(shape=(512), dtype=float64)
    l_step = cuda.shared.array(shape=(512), dtype=int32)

    noise[i] = random.xoroshiro128p_normal_float64(rng_states, j)

    if j < x.shape[0]:
        l_x[i] = x[j]
        l_px[i] = px[j]
        l_step[i] = step_values[j]
        for k in range(n_iterations):
            action[i] = (l_x[i] * l_x[i] + l_px[i] * l_px[i]) * 0.5
            rot_angle[i] = omega_0 + (omega_1 + action[i]) + \
                (0.5 * omega_2 * action[i] * action[i])

            if (l_x[i] == 0.0 and l_px[i] == 0.0) or action[i] >= action_radius:
                l_x[i] = 0.0
                l_px[i] = 0.0
                break
            
            temp1[i] = l_x[i]
            temp2[i] = (
                l_px[i] + epsilon * noise[i] * (l_x[i] ** beta)
                * math.exp(-((x_star / (delta + abs(l_x[i]))) ** alpha))
            )
            l_x[i] = math.cos(rot_angle[i]) * temp1[i] + \
                math.sin(rot_angle[i]) * temp2[i]
            l_px[i] = -math.sin(rot_angle[i]) * temp1[i] + \
                math.cos(rot_angle[i]) * temp2[i]

            l_step[i] += 1

            noise[i] = random.xoroshiro128p_normal_float64(rng_states, j) + gamma * noise[i]
        x[j] = l_x[i]
        px[j] = l_px[i]
        step_values[j] = l_step[i]
