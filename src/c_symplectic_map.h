#ifndef C_SYMPLECTIC_MAP_H
#define C_SYMPLECTIC_MAP_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>

#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
#include <cuda.h>
#endif

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>

// symplectic map functor

struct functor_map
{
    thrust::minstd_rand rng;

    double omega_0, omega_1, omega_2;
    double epsilon;
    double x_star, delta;
    double alpha, beta;
    
    double barrier_radius, action_radius;
    
    bool common, correlated;

    // common noise
    thrust::device_vector<double> noise_pointer;
    unsigned int n_iterations;

    // correlated not common noise
    double gamma;

    // Constructors

    functor_map(double _omega_0, double _omega_1, double _omega_2, double _epsilon, double _x_star, double _delta, double _alpha, double _beta, double _barrier_radius, thrust::device_vector<double> _noise_pointer, unsigned int _n_iterations); // common noise
    functor_map(double _omega_0, double _omega_1, double _omega_2, double _epsilon, double _x_star, double _delta, double _alpha, double _beta, double _barrier_radius, unsigned int _n_iterations, double _gamma); // correlated not common noise

    // Functor Operator

    template <typename Tuple> __host__ __device__ void operator()(Tuple t);

    // Methods

    __host__ __device__ thrust::device_vector<double> make_correlated_noise(unsigned int size, double gamma, double starting_point);
};

// symplectic_map_struct

struct symplectic_map
{
    double omega_0, omega_1, omega_2;
    double epsilon;
    double x_star, delta;
    double alpha, beta;
    
    double barrier_radius;

    thrust::device_vector<double> noise;

    thrust::device_vector<double> X, P;
    thrust::device_vector<double> X_0, P_0;
    thrust::device_vector<unsigned int> T;
    thrust::device_vector<double> last_hit;

    // Constructor

    symplectic_map(double _omega_0, double _omega_1, double _omega_2, double _epsilon, double _x_star, double _delta, double _alpha, double _beta, double _barrier_radius, std::vector<double> _X_0, std::vector<double> _P_0);

    // Methods
    
    void reset();
    void compute(unsigned int kernel_iterations, unsigned int block_iterations, double gamma=0.0);
    void compute(unsigned int kernel_iterations, unsigned int block_iterations, std::vector<double> given_noise);

    // Getters (to be binded)

    std::vector<double> x();
    std::vector<double> p();
    std::vector<double> x0();
    std::vector<double> p0();
    std::vector<unsigned int> t();
};

#endif // C_SYMPLECTIC_MAP_H