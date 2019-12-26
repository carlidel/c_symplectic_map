#include "c_symplectic_map.h"

// symplectic map functor

functor_map::functor_map(double _omega_0, double _omega_1, double _omega_2, double _epsilon, double _x_star, double _delta, double _alpha, double _beta, double _barrier_radius, thrust::device_vector<double> _noise_pointer, unsigned int _n_iterations) : omega_0(_omega_0), omega_1(_omega_1), omega_2(_omega_2), epsilon(_epsilon), x_star(_x_star), delta(_delta), alpha(_alpha), beta(_beta), barrier_radius(_barrier_radius), noise_pointer(_noise_pointer), n_iterations(_n_iterations)
{
    // Set switches
    common = true;
    correlated = false;
    // Set corresponding barrier
    action_radius = barrier_radius * barrier_radius * 0.5;
}

functor_map::functor_map(double _omega_0, double _omega_1, double _omega_2, double _epsilon, double _x_star, double _delta, double _alpha, double _beta, double _barrier_radius, unsigned int _n_iterations, double _gamma) : omega_0(_omega_0), omega_1(_omega_1), omega_2(_omega_2), epsilon(_epsilon), x_star(_x_star), delta(_delta), alpha(_alpha), beta(_beta), barrier_radius(_barrier_radius), n_iterations(_n_iterations), gamma(_gamma)
{
    // Set switches
    common = false;
    correlated = true;
    // Set corresponding barrier
    action_radius = barrier_radius * barrier_radius * 0.5;
}


template <typename Tuple> __host__ __device__ void functor_map::operator()(Tuple t)
{
    double action, rot_angle;
    double temp1, temp2;

    thrust::device_vector <double> personal_noise;
    if(correlated)
    {
        personal_noise = make_correlated_noise(n_iterations, gamma, thrust::get<3>(t));
    }

    for (unsigned int i = 0; i < n_iterations; i++)
    {
        // Initialize action and rotation values
        action = ((thrust::get<0>(t) * thrust::get<0>(t)) + (thrust::get<1>(t) * thrust::get<1>(t))) * 0.5;
        rot_angle = omega_0 + (omega_1 * action) + (omega_2 * action * action * 0.5);
        
        // Is the particle lost?
        if ((thrust::get<0>(t) == 0.0 && thrust::get<1>(t) == 0.0) || action >= action_radius)
        {
            thrust::get<0>(t) = 0;
            thrust::get<1>(t) = 0;
            return;
        }

        // second vector element
        temp1 = thrust::get<0>(t);
        temp2 = thrust::get<1>(t) 
            + epsilon 
                * (common ? noise_pointer[i] : personal_noise[i]) 
                * pow(abs(thrust::get<0>(t)), beta)
                * exp(-pow((x_star / (delta + abs(thrust::get<0>(t)))), alpha));
        
        thrust::get<0>(t) = cos(rot_angle) * temp1 + sin(rot_angle) * temp2;
        thrust::get<1>(t) = -sin(rot_angle) * temp1 + cos(rot_angle) * temp2;

        thrust::get<2>(t) += 1; 
    }

    if(correlated)
    {
        thrust::get<3>(t) = personal_noise[n_iterations - 1]; 
    }
    
    return;
}

__host__ __device__ thrust::device_vector<double> functor_map::make_correlated_noise(unsigned int size, double gamma, double starting_point)
{
    thrust::random::normal_distribution<double> dist(0.0, 1.0);
    thrust::device_vector<double> noise(size);

    for(unsigned int i = 0; i < size; i++)
        noise[i] = dist(rng);

    if(gamma != 0.0)
        for (unsigned int i = 1; i < size; i++)
            noise[i] += gamma * noise[i - 1];

    return noise;
}

// symplectic_map_struct

c_symplectic_map::c_symplectic_map(double _omega_0, double _omega_1, double _omega_2, double _epsilon, double _x_star, double _delta, double _alpha, double _beta, double _barrier_radius, std::vector<double> _X_0, std::vector<double> _P_0) : omega_0(_omega_0), omega_1(_omega_1), omega_2(_omega_2), epsilon(_epsilon), x_star(_x_star), delta(_delta), alpha(_alpha), beta(_beta), barrier_radius(_barrier_radius), X(_X_0), P(_P_0), X_0(_X_0), P_0(_P_0)
{
    T.resize(X_0.size());
    thrust::fill(T.begin(), T.end(), 0);
    last_hit.resize(T.size());
}

void c_symplectic_map::reset()
{
    X = X_0;
    P = P_0;
    thrust::fill(T.begin(), T.end(), 0);
}

void c_symplectic_map::compute(unsigned int kernel_iterations, unsigned int block_iterations, double gamma)
{
    functor_map func(omega_0, omega_1, omega_2, epsilon, x_star, delta, alpha, beta, barrier_radius, kernel_iterations, gamma);
    thrust::fill(last_hit.begin(), last_hit.end(), 0.0);

    for(unsigned int i = 0; i < block_iterations; i++)
    {
        thrust::for_each
        (
            thrust::make_zip_iterator(thrust::make_tuple(X.begin(), P.begin(), T.begin(), last_hit.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(X.end(), P.end(), T.end(), last_hit.end())),
            func
        );
    }
}

void c_symplectic_map::compute(unsigned int kernel_iterations, unsigned int block_iterations, std::vector<double> given_noise)
{
    thrust::device_vector<double> noise(kernel_iterations * block_iterations);
    thrust::copy(given_noise.begin(), given_noise.end(), noise.begin());

    thrust::device_vector<double> temp(kernel_iterations);

    for (unsigned int i = 0; i < block_iterations; i++)
    {
        thrust::copy(noise.begin() + (i * kernel_iterations), noise.begin() + ((i + 1) * kernel_iterations), temp.begin());
        functor_map func(omega_0, omega_1, omega_2, epsilon, x_star, delta, alpha, beta, barrier_radius, temp, kernel_iterations);
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(X.begin(), P.begin(), T.begin(), last_hit.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(X.end(), P.end(), T.end(), last_hit.end())),
            func);
    }
}

// Getters

std::vector<double> c_symplectic_map::x()
{
    std::vector<double> X_copy(X.size());
    thrust::copy(X.begin(), X.end(), X_copy.begin());
    return X_copy;
}

std::vector<double> c_symplectic_map::p()
{
    std::vector<double> P_copy(P.size());
    thrust::copy(P.begin(), P.end(), P_copy.begin());
    return P_copy;
}

std::vector<double> c_symplectic_map::x0()
{
    std::vector<double> X_0_copy(X_0.size());
    thrust::copy(X_0.begin(), X_0.end(), X_0_copy.begin());
    return X_0_copy;
}

std::vector<double> c_symplectic_map::p0()
{
    std::vector<double> P_0_copy(P_0.size());
    thrust::copy(P_0.begin(), P_0.end(), P_0_copy.begin());
    return P_0_copy;
}

std::vector<unsigned int> c_symplectic_map::t()
{
    std::vector<unsigned int> T_copy(T.size());
    thrust::copy(T.begin(), T.end(), T_copy.begin());
    return T_copy;
}