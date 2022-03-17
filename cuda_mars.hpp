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

double free_memory_size()
{
    int num_gpus;
    size_t free, total;
    cudaGetDeviceCount( &num_gpus );
    for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
        cudaSetDevice( gpu_id );
        int id;
        cudaGetDevice( &id );
        cudaMemGetInfo( &free, &total );
        return free/1e9; // converting to GB
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void __global__ randoms_to_range_kernel(T *_data, int _size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _size)
    {
        _data[idx] = (_data[idx] - 0.5)*2; // [0, 1) into [-1, 1)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void __global__ process_large_matrix_kernel(T *_data, int _size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _size)
    {
        _data[idx] = (_data[idx] - 0.5)*2; // [0, 1) into [-1, 1)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE 1024

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void gpu_fill_rand(double *_data, int _size)
{
    curandGenerator_t randGen;
    curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniformDouble(randGen, _data, _size);
    SAFE_KERNEL_CALL((randoms_to_range_kernel<<<(_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(_data, _size)));
}

void gpu_fill_rand(float *_data, int _size)
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

#define VWARP_SIZE 32
#define VWARP_NUM 32

template <typename _T>
__device__ __forceinline__ _T warp_reduce_sum(_T val)
{
    for (int offset = VWARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
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

template <typename _T>
__forceinline__ __device__ _T block_reduce_shmem(_T val, int tid)
{
    __shared__ _T sdata[BLOCK_SIZE];

    sdata[tid] = val;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    return sdata[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void mars_mc_block_per_i_kernel(T* _mat,
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

            for(int i = 0; i < _size; i++)
            {
                T val = 0;
                int offset = tid;
                while(offset < _size)
                {
                    val += _mat[i*_size + offset] * _spins[offset + block_id * _size];
                    offset += blockDim.x;
                }
                T sum = block_reduce_sum(val);

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
        } while(d[0] >= _d_min);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void mars_mc_warp_per_i_kernel(const T* __restrict__ _mat,
                                          T* _spins,
                                          const T * __restrict__ _h,
                                          int _size,
                                          T _c_step,
                                          T _d_min,
                                          T _alpha,
                                          const T * __restrict__ _tempratures)
{
    int block_id = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / VWARP_SIZE;
    int lane_id = tid % VWARP_SIZE;

    int par_shift = block_id*VWARP_NUM + warp_id;

    __shared__ T current_temperature[VWARP_NUM];
    __shared__ T d[VWARP_NUM];
    current_temperature[warp_id] = _tempratures[par_shift];

    __syncwarp();

    while(current_temperature[warp_id] > 0)
    {
        __syncwarp();
        if(lane_id == 0)
        {
            current_temperature[warp_id] -= _c_step;
        }

        do
        {
            __syncwarp();
            if(lane_id == 0)
                d[warp_id] = 0;
            __syncwarp();

            for(int i = 0; i < _size; i++)
            {
                T val = 0;
                #pragma unroll(16)
                for(int offset = lane_id; offset < _size; offset += VWARP_SIZE)
                {
                    val += _mat[i*_size + offset] * _spins[par_shift * _size + offset];
                }
                T sum = warp_reduce_sum(val);
                T mean_field = sum + _h[i];

                if(lane_id == 0)
                {
                    T s_trial = 0;

                    if(current_temperature[warp_id] > 0)
                    {
                        s_trial = _alpha * (-tanhf(mean_field / current_temperature[warp_id])) + (1 - _alpha) * _spins[i + par_shift * _size];
                    }
                    else if (mean_field > 0)
                        s_trial = -1;
                    else
                        s_trial = 1;

                    T abs_val = fabsf(s_trial - _spins[i + par_shift * _size]);
                    if(abs_val > d[warp_id])
                    {
                        d[warp_id] = abs_val;
                    }
                    _spins[i + par_shift * _size] = s_trial;
                }
            }
            __syncwarp();
        } while(d[warp_id] >= _d_min);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ T dot_product(T *v1, T* v2, int _size)
{
    int tid = threadIdx.x;

    T val = 0;
    int offset = tid;
    while(offset < _size)
    {
        val += v1[offset]*v2[offset];
        offset += blockDim.x;
    }
    T dot = block_reduce_sum(val);
    return dot;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ T dot_product_mxv(T* _mat, T* _spin, int _size)
{
    int tid = threadIdx.x;

    T val = 0;
    int offset = tid;
    while(offset < _size)
    {
        T vxm_val = 0;
        for(int i = 0; i < _size; i++)
        {
            vxm_val += _spin[i]*_mat[i*_size + offset];
        }
        val += _spin[offset] * vxm_val;
        offset += blockDim.x;
    }

    T dot = block_reduce_sum(val);
    return dot;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMin(double* address, double val)
{
    unsigned long long* address_as_i = (unsigned long long*) address;
    unsigned long long old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __double_as_longlong(::fminf(val, __longlong_as_double (assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void estimate_min_energy_kernel(T* _mat,
                                            T *_spins,
                                            T *_h,
                                            int _size,
                                            int _num_iters,
                                            T *_min_energy)
{
    int tid = threadIdx.x;
    int block_id = blockIdx.x;

    T* cur_spin = &_spins[_size * block_id];

    T energy = dot_product(_h, cur_spin, _size) + dot_product_mxv(_mat, cur_spin, _size);
    atomicMin(&_min_energy[0], energy);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int round_up(int numToRound, int multiple)
{
    if (multiple == 0)
        return numToRound;

    int remainder = numToRound % multiple;
    if (remainder == 0)
        return numToRound;

    return numToRound + multiple - remainder;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
auto cuda_mars(SquareMatrix<T> &_J_mat,
               std::vector<T> &_h,
               int _n,
               int _t_min,
               int _t_max,
               T _c_step,
               T _d_min,
               T _alpha,
               T _t_step,
               double &_time)
{
    double free_mem = 0.5/*not to waste all*/*free_memory_size();
    int max_blocks_mem_fit = (free_mem*1024*1024*1024 - _n*_n*sizeof(T))/ (_n *sizeof(T));
    std::cout << "we can simultaneously store " << max_blocks_mem_fit << " spins in " << free_mem << " GB of available memory" << std::endl;

    int num_steps = round_up((_t_max - _t_min) / _t_step, VWARP_NUM);
    std::cout << "number of temperatures steps: " << num_steps << std::endl;
    std::cout << "matrix size: " << _n << std::endl;
    int block_size = 1024;
    int num_blocks = min(num_steps, max_blocks_mem_fit)/VWARP_NUM;
    std::cout << "estimated block size: " << block_size << std::endl;
    std::cout << "estimated number of blocks: " << num_blocks << std::endl;
    std::cout << "we will do  " << num_blocks * VWARP_NUM << " MC steps in parallel" << std::endl;

    std::cout << "Using CUDA mars (parallelism for different MC steps)" << std::endl;
    T *dev_s, *dev_h, *dev_temperatures;
    T *min_energy;
    SAFE_CALL(cudaMallocManaged((void**)&dev_s, VWARP_NUM*_n*num_blocks*sizeof(T)));
    SAFE_CALL(cudaMallocManaged((void**)&dev_h, _n*sizeof(T)));
    SAFE_CALL(cudaMallocManaged((void**)&dev_temperatures, VWARP_NUM*num_blocks*sizeof(T)));
    SAFE_CALL(cudaMallocManaged((void**)&min_energy, sizeof(T)));
    min_energy[0] = std::numeric_limits<T>::max();

    T *dev_mat;
    SAFE_CALL(cudaMallocManaged((void**)&dev_mat, _n*_n*sizeof(T)));
    SAFE_CALL(cudaMemcpy(dev_mat, _J_mat.get_ptr(), _n*_n*sizeof(T), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dev_h, &(_h[0]), _n*sizeof(T), cudaMemcpyHostToDevice));

    std::cout << "using warp per i policy" << std::endl;
    double t1 = omp_get_wtime();
    for(base_type temperature = _t_min; temperature < _t_max; temperature += (_t_step * num_blocks * VWARP_NUM))
    {
        gpu_fill_rand(dev_s, _n*num_blocks*VWARP_NUM);

        for(int i = 0; i < num_blocks*VWARP_NUM; i++)
            dev_temperatures[i] = temperature + _t_step*i;

        /*SAFE_KERNEL_CALL((mars_mc_block_per_i_kernel<<<num_blocks , block_size>>>(dev_mat,
                                 dev_s, dev_h, _n, _c_step, _d_min, _alpha, dev_temperatures)));*/

        SAFE_KERNEL_CALL((mars_mc_warp_per_i_kernel<<<num_blocks , block_size>>>(dev_mat,
                                 dev_s, dev_h, _n, _c_step, _d_min, _alpha, dev_temperatures)));

        SAFE_KERNEL_CALL((estimate_min_energy_kernel<<<VWARP_NUM*num_blocks, 32>>>(dev_mat,
                                    dev_s, dev_h, _n, num_steps, min_energy)));
    }
    double t2 = omp_get_wtime();
    std::cout << "CUDA calculations finished in " << (t2 - t1) << " seconds" << std::endl;
    std::cout << "CUDA min energy: " << std::setprecision(10) << min_energy[0] << std::endl;
    _time = t2 - t1;

    std::vector<T> result(_n);

    SAFE_CALL(cudaMemcpy(&result[0], dev_s, sizeof(T)*_n, cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaFree(dev_s));
    SAFE_CALL(cudaFree(dev_mat));
    SAFE_CALL(cudaFree(dev_h));
    SAFE_CALL(cudaFree(dev_temperatures));
    T min_energy_val = min_energy[0];
    SAFE_CALL(cudaFree(min_energy));

    return min_energy_val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
