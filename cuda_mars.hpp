#pragma once
#include <iostream>
#include <curand.h>
#include "safe_calls.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void __global__ randoms_to_range_kernel(T *_data, size_t _size)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _size)
    {
        _data[idx] = (_data[idx] - 0.5)*2; // [0, 1) into [-1, 1)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void __global__ process_large_matrix_kernel(T *_data, size_t _size)
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
    SAFE_KERNEL_CALL((randoms_to_range_kernel<<<(_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_data, _size)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void mars_mc_parallel_kernel(float* _mat,
                                        float* _spins,
                                        int _size,
                                        float* _temp,
                                        float _temp_step,
                                        float* _phi,
                                        bool* _continue_iteration,
                                        float _min_diff,
                                        float _alpha)
{
    int blockId = blockIdx.x;
    int tid = threadIdx.x;

    do
    {
        // Lessen temperature
        if (tid == 0)
            _temp[blockId] = temp[blockId] - _temp_step;

        // Stabilize
        do
        {
            __syncthreads();
            
            // By default current iteration is the last one
            if (tid == 0)
                _continue_iteration[blockId] = false;

            for (int spinId = 0; spinId < size; ++spinId)
            {
                __syncthreads();

                // Transitional value assignment
                int wIndex = tid;
                while (wIndex < _size)
                {
                    _phi[wIndex + blockId * size] =
                            spins[_size + blockId * size] * mat[spinId * size + _size];

                    wIndex = wIndex + blockDim.x;
                }
                __syncthreads();

                // Parallelized mean-field computation
                long long offset = 1;
                while (offset < _size)
                {
                    wIndex = tid;
                    while ((wIndex * 2 + 1) * offset < _size)
                    {
                        _phi[wIndex * 2 * offset + blockId * size] += _phi[(wIndex * 2 + 1) * offset
                                                                           + blockId * size];
                        wIndex = wIndex + blockDim.x;
                    }
                    offset *= 2;
                    __syncthreads();
                }
                __syncthreads();

                // Mean-field calculation complete - write new spin and delta
                if (tid == 0) 
                {
                    float meanField = _phi[blockId * size];
                    float old = spins[spinId + blockId * size];
                    if (_temp[blockId] > 0)
                    {
                        spins[spinId + blockId * size] = -1 * tanh(meanField / _temp[blockId]) * linearCoef
                                     + spins[spinId + blockId * size] * (1 - linearCoef);
                    }
                    else if (meanField > 0)
                        spins[spinId + blockId * size] = -1;
                    else
                        spins[spinId + blockId * size] = 1;

                    if (_min_diff < fabs(old - spins[spinId + blockId * size]))
                        _continue_iteration[blockId] = true; // Too big delta. One more iteration needed
                }
                __syncthreads();
            }
        } while (_continue_iteration[blockId]);
    } while (_temp[blockId] >= 0);
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
    T *dev_s, *dev_s_trial, *dev_phi;
    bool *dev_continue_iteration;
    T* dev_temp;
    SAVE_CALL(cudaMallocManaged((void**)&s, _n*sizeof(T)));
    SAVE_CALL(cudaMallocManaged((void**)&s_trial, _n*sizeof(T)));
    SAVE_CALL(cudaMallocManaged((void**)&phi, _n*sizeof(T)));
    SAVE_CALL(cudaMallocManaged((void**)&dev_continue_iteration, sizeof(bool)));
    SAVE_CALL(cudaMallocManaged((void**)&dev_temp, sizeof(T)));

    T *dev_mat;
    SAVE_CALL(cudaMallocManaged((void**)&dev_mat, _n*_n*sizeof(T)));
    SAVE_CALL(cudaMemcpy(dev_mat, _J_mat.get_ptr(), _n*_n*sizeof(T), cudaMemcpyHostToDevice));

    T current_temperature = 0;
    for(base_type temperature = _t_min; temperature < _t_max; temperature += _t_step)
    {
        gpu_fill_rand(dev_s, _n);

        current_temperature = temperature; // t' = t

        SAFE_KERNEL_CALL((mars_mc_parallel_kernel<<<1, min(BLOCK_SIZE, _n)>>>(dev_mat,
                                 dev_s, _n, dev_temp, _c_step, dev_phi, dev_continue_iteration, _d_min, _alpha)));
    }
    std::cout << "done!" << std::endl;

    std::vector<T> result(_n);

    SAVE_CALL(cudaMemcpy(&result[0], s, sizeof(T)*_n, cudaMemcpyDeviceToHost));
    SAVE_CALL(cudaFree(dev_s_trial));
    SAVE_CALL(cudaFree(dev_phi));
    SAVE_CALL(cudaFree(dev_s));
    SAVE_CALL(cudaFree(dev_mat));
    SAVE_CALL(cudaFree(dev_temp));
    SAVE_CALL(cudaFree(dev_continue_iteration));

    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
