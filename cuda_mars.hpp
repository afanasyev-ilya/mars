#pragma once
#include <iostream>
#include <curand.h>
#include "safe_calls.hpp"
#include <omp.h>

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

void gpu_fill_rand(float *_data, size_t _size)
{
    curandGenerator_t randGen;
    curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(randGen, _data, _size);
    SAFE_KERNEL_CALL((randoms_to_range_kernel<<<(_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_data, _size)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void mars_mc_parallel_kernel(T* _mat,
                                        T* _spins,
                                        T *_h,
                                        int _size,
                                        T* _temp,
                                        T _c_step,
                                        T* _phi,
                                        bool* _continue_iteration,
                                        T _d_min,
                                        T _alpha)
{
    int block_id = blockIdx.x;
    int tid = threadIdx.x;

    /*__shared__ int continue_iteration[1];

    do
    {
        // Lessen temperature
        if (tid == 0)
            _temp[block_id] = _temp[block_id] - _temp_step;

        // Stabilize
        do
        {
            __syncthreads();
            
            // By default current iteration is the last one
            if (tid == 0)
                continue_iteration[0] = false;

            for (int spin_id = 0; spin_id < _size; ++spin_id)
            {
                __syncthreads();

                if(tid == 0)
                {
                    T reduction_result = 0;
                    for(size_t j = 0; j < _size; j++)
                    {
                        reduction_result += _mat[_size * spin_id + j] * _spins[block_id * _size + j];
                    }
                    _phi[block_id * _size + spin_id] = reduction_result + _h[spin_id];
                }

                __syncthreads();

                // Mean-field calculation complete - write new spin and delta
                if (tid == 0) 
                {
                    T mean_field = _phi[block_id * _size + spin_id];
                    T old = _spins[spin_id + block_id * _size];
                    if (_temp[block_id] > 0)
                    {
                        _spins[spin_id + block_id * _size] = -1 * tanh(mean_field / _temp[block_id]) * _alpha
                                     + _spins[spin_id + block_id * _size] * (1 - _alpha);
                    }
                    else if (mean_field > 0)
                        _spins[spin_id + block_id * _size] = -1;
                    else
                        _spins[spin_id + block_id * _size] = 1;

                    if (fabs(old - _spins[spin_id + block_id * _size]) > _min_diff)
                        continue_iteration[0] = true; // Too big delta. One more iteration needed
                }
                __syncthreads();
            }
            if (tid == 0)
                printf("cont = %d\n", continue_iteration[0]);
        } while (continue_iteration[0]);
    } while (_temp[block_id] >= 0);*/

    if(tid == 0)
    {
        T current_temperature = _temp[block_id];

        while(current_temperature > 0)
        {
            T d = 0;
            current_temperature -= _c_step;

            do
            {
                for(size_t i = 0; i < _size; i++)
                {
                    T sum = 0;
                    for(size_t j = 0; j < _size; j++)
                    {
                        sum += _mat[i*_size + j] * _spins[j];
                    }
                    _phi[i] = sum + _h[i];

                    T s_trial = 0;

                    if(current_temperature > 0)
                    {
                        s_trial = _alpha * (-tanh(_phi[i] / current_temperature)) + (1 - _alpha) * _spins[i];
                    }
                    else if (_phi[i] > 0)
                        s_trial = -1;
                    else
                        s_trial = 1;

                    if(fabs(s_trial - _spins[i]) > d)
                    {
                        d = abs(s_trial - _spins[i]);
                    }

                    _spins[i] = s_trial;
                }
            } while(d < _d_min);
        }
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
    std::cout << "Using CUDA mars (parallelism for different MC steps)" << std::endl;
    T *dev_s, *dev_s_trial, *dev_phi, *dev_h;
    bool *dev_continue_iteration;
    T* dev_temp;
    SAFE_CALL(cudaMallocManaged((void**)&dev_s, _n*sizeof(T)));
    SAFE_CALL(cudaMallocManaged((void**)&dev_s_trial, _n*sizeof(T)));
    SAFE_CALL(cudaMallocManaged((void**)&dev_phi, _n*sizeof(T)));
    SAFE_CALL(cudaMallocManaged((void**)&dev_continue_iteration, sizeof(bool)));
    SAFE_CALL(cudaMallocManaged((void**)&dev_temp, sizeof(T)));
    SAFE_CALL(cudaMallocManaged((void**)&dev_h, _n*sizeof(T)));

    T *dev_mat;
    SAFE_CALL(cudaMallocManaged((void**)&dev_mat, _n*_n*sizeof(T)));
    SAFE_CALL(cudaMemcpy(dev_mat, _J_mat.get_ptr(), _n*_n*sizeof(T), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dev_h, &(_h[0]), _n*sizeof(T), cudaMemcpyHostToDevice));

    T current_temperature = 0;
    double t1 = omp_get_wtime();
    for(base_type temperature = _t_min; temperature < _t_max; temperature += (_t_step * NUM_BLOCKS))
    {
        gpu_fill_rand(dev_s, _n);

        current_temperature = temperature; // t' = t

        for(int i = 0; i < NUM_BLOCKS; i++)
            dev_temp[i] = temperature + _t_step*i;

        int block_size = min((size_t)BLOCK_SIZE, (size_t)_n);
        SAFE_KERNEL_CALL((mars_mc_parallel_kernel<<<NUM_BLOCKS, block_size>>>(dev_mat,
                                 dev_s, dev_h, _n, dev_temp, _c_step, dev_phi, dev_continue_iteration, _d_min, _alpha)));
        std::cout << "block portion done" << std::endl;
    }
    double t2 = omp_get_wtime();
    std::cout << "GPU calculations finished in " << (t2 - t1) << " seconds" << std::endl;

    std::vector<T> result(_n);

    SAFE_CALL(cudaMemcpy(&result[0], dev_s, sizeof(T)*_n, cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaFree(dev_s_trial));
    SAFE_CALL(cudaFree(dev_phi));
    SAFE_CALL(cudaFree(dev_s));
    SAFE_CALL(cudaFree(dev_mat));
    SAFE_CALL(cudaFree(dev_temp));
    SAFE_CALL(cudaFree(dev_continue_iteration));
    SAFE_CALL(cudaFree(dev_h));

    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
