#pragma once
#include <iostream>
#include <curand.h>
#include "safe_calls.hpp"
#include <omp.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned warp_id()
{
    // this is not equal to threadIdx.x / 32
    unsigned ret;
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if __CUDA_ARCH__ >= 700
#define FULL_MASK 0xffffffff
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
static __device__ __forceinline__ _T shfl_down( _T r, int offset )
{
#if __CUDA_ARCH__ >= 700
    return __shfl_down_sync(FULL_MASK, r, offset );
#elif __CUDA_ARCH__ >= 300
    return __shfl_down( r, offset );
#else
    return 0.0f;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
__inline__ __device__ _T warp_reduce_sum(_T val)
{
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += shfl_down(val, offset);
    return val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
__inline__ __device__ _T block_reduce_sum(_T val)
{
    static __shared__ _T shared[32]; // Shared mem for 32 partial sums
    int lane =  threadIdx.x % 32;
    int wid =  threadIdx.x / 32;

    val = warp_reduce_sum(val);     // Each warp performs partial reduction

    if (lane==0)
        shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if(wid == 0)
        val = warp_reduce_sum(val); //Final reduce within first warp

    return val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void mars_mc_parallel_kernel(T* _mat,
                                        T* _spins,
                                        T *_h,
                                        int _size,
                                        T _c_step,
                                        T _d_min,
                                        T _alpha,
                                        T *_tempratures)
{
    int block_id = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ T current_temperature[1];
    __shared__ T d[1];
    current_temperature[0] = _tempratures[block_id];

    __syncthreads();

    while(current_temperature[0] > 0)
    {
        __syncthreads();
        if(tid == 0)
        {
            current_temperature[0] -= _c_step;
        }

        do
        {
            __syncthreads();
            if(tid == 0)
                d[0] = 0;
            __syncthreads();

            for(size_t i = 0; i < _size; i++)
            {
                T sum = 0;
                for(size_t j = 0; j < _size; j++)
                {
                    sum += _mat[i*_size + j] * _spins[j + block_id * _size];
                }

                /*T new_sum = block_reduce_sum(_mat[i*_size + tid] * _spins[tid + block_id * _size]);

                if(tid == 0)
                {
                    if(sum != new_sum)
                        printf("%lf %lf error!\n", sum, new_sum);
                    else
                        printf("%lf %lf correct!\n", sum, new_sum);
                }*/

                T mean_field = sum + _h[i];

                if(tid == 0)
                {
                    T s_trial = 0;

                    if(current_temperature[0] > 0)
                    {
                        s_trial = _alpha * (-tanh(mean_field / current_temperature[0])) + (1 - _alpha) * _spins[i + block_id * _size];
                    }
                    else if (mean_field > 0)
                        s_trial = -1;
                    else
                        s_trial = 1;

                    if(fabs(s_trial - _spins[i + block_id * _size]) > d[0])
                    {
                        d[0] = fabs(s_trial - _spins[i + block_id * _size]);
                    }
                    _spins[i + block_id * _size] = s_trial;
                }
            }
            __syncthreads();
        } while(d[0] < _d_min);
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
    size_t max_blocks_mem_fit = (MEM_SIZE*1024*1024 - _n*_n*sizeof(T))/ (_n *sizeof(T));
    std::cout << "we can store " << max_blocks_mem_fit << " spins in " << MEM_SIZE << " GB of available memory" << std::endl;

    size_t num_steps = (_t_max - _t_min) / _t_step;
    std::cout << "number of temperatures steps: " << num_steps << std::endl;
    std::cout << "matrix size: " << _n << std::endl;
    int block_size = min((size_t)BLOCK_SIZE, _n);
    int num_blocks = min(num_steps, max_blocks_mem_fit);
    std::cout << "estimated block size: " << block_size << std::endl;
    std::cout << "estimated number of blocks: " << num_blocks << std::endl;

    std::cout << "Using CUDA mars (parallelism for different MC steps)" << std::endl;
    T *dev_s, *dev_h, *dev_temperatures;
    SAFE_CALL(cudaMallocManaged((void**)&dev_s, _n*num_blocks*sizeof(T)));
    SAFE_CALL(cudaMallocManaged((void**)&dev_h, _n*sizeof(T)));
    SAFE_CALL(cudaMallocManaged((void**)&dev_temperatures, num_blocks*sizeof(T)));

    T *dev_mat;
    SAFE_CALL(cudaMallocManaged((void**)&dev_mat, _n*_n*sizeof(T)));
    SAFE_CALL(cudaMemcpy(dev_mat, _J_mat.get_ptr(), _n*_n*sizeof(T), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dev_h, &(_h[0]), _n*sizeof(T), cudaMemcpyHostToDevice));

    double t1 = omp_get_wtime();
    for(base_type temperature = _t_min; temperature < _t_max; temperature += (_t_step * num_blocks))
    {
        gpu_fill_rand(dev_s, _n*num_blocks);

        for(int i = 0; i < num_blocks; i++)
            dev_temperatures[i] = temperature + _t_step*i;

        SAFE_KERNEL_CALL((mars_mc_parallel_kernel<<<num_blocks , block_size>>>(dev_mat,
                                 dev_s, dev_h, _n, _c_step, _d_min, _alpha, dev_temperatures)));
    }
    double t2 = omp_get_wtime();
    std::cout << "GPU calculations finished in " << (t2 - t1) << " seconds" << std::endl;

    std::vector<T> result(_n);

    SAFE_CALL(cudaMemcpy(&result[0], dev_s, sizeof(T)*_n, cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaFree(dev_s));
    SAFE_CALL(cudaFree(dev_mat));
    SAFE_CALL(cudaFree(dev_h));
    SAFE_CALL(cudaFree(dev_temperatures));

    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
