#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void gpu_fill_rand(T *_data, size_t _size)
{
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    curandGenerateUniform(prng, _data, _size);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
auto cuda_mars(SquareMatrix<T> &_J_mat,
               std::vector<T> &_h,
               size_t _n,
               int _t_min,
               int _t_max,
               T _c_step,
               T _d_min,
               T _alpha,
               T _t_step)
{
    thrust::device_vector<T> s(_n);
    thrust::device_vector<T> s_new(_n);

    T current_temperature = 0;
    for(base_type temperature = _t_min; temperature < _t_max; temperature += _t_step)
    {
        GPU_fill_rand(thrust::raw_pointer_cast(&s[0]), s.size());
        for(int i = 0; i < 3; i++)
            std::cout << s[i] << " ";
        std::cout << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
