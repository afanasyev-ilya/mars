#pragma once
#include <iostream>
#include <curand.h>
#include "safe_calls.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void __global__ randoms_to_range(T *_data, size_t _size)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _size)
    {
        _data[idx] = (_data[idx] - 0.5)*2; // [0, 1) into [-1, 1)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE 1024

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void gpu_fill_rand(double *_data, size_t _size)
{
    curandGenerator_t randGen;
    curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniformDouble(randGen, _data, _size);
    SAFE_KERNEL_CALL((randoms_to_range<<<(_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_data, _size)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void cpu_fill_rand(double *_data, size_t _size)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> uni(-1, 1);

    for (size_t i = 0; i < _size; i++)
    {
        _data[i] = uni(rng);
    }
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
    T *s, *s_new, *phi;
    cudaMallocManaged((void**)&s, _n*sizeof(T));
    cudaMallocManaged((void**)&s_new, _n*sizeof(T));
    cudaMallocManaged((void**)&phi, _n*sizeof(T));

    T current_temperature = 0;
    for(base_type temperature = _t_min; temperature < _t_max; temperature += _t_step)
    {
        cpu_fill_rand(s, _n);

        current_temperature = temperature; // t' = t

        while(current_temperature > 0)
        {
            T d = 0;
            current_temperature -= _c_step;
            do
            {
                for(size_t i = 0; i < _n; i++)
                {
                    T sum = 0;
                    for(size_t j = 0; j < _n; j++)
                    {
                        sum += _J_mat.get(i, j) * s[j];
                    }
                    phi[i] = sum + _h[i];

                    if(current_temperature > 0)
                    {
                        s_new[i] = _alpha * (-tanh(phi[i] / current_temperature)) + (1 - _alpha) * s[i];
                    }
                    else
                    {
                        s_new[i] = -sign(phi[i]);
                    }

                    if(abs(s_new[i] - s[i]) > d)
                    {
                        d = abs(s_new[i] - s[i]);
                    }

                    s[i] = s_new[i];
                }
            } while(d < _d_min);
        }
    }

    std::vector<T> result(_n);
    cudaMemcpy(&result[0], s, sizeof(T)*_n, cudaMemcpyDeviceToHost);
    cudaFree(s_new);
    cudaFree(phi);
    cudaFree(s);

    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
