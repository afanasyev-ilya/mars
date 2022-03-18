/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define FULL_MASK 0xffffffff

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define VWARP_SIZE 8
#define VWARP_NUM 128
#define BLOCK_SIZE 1024

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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

template <typename _T>
__device__ __forceinline__ _T warp_reduce_sum(_T val)
{
    for (int offset = 32/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
__device__ __forceinline__ _T virt_warp_reduce_sum(_T val)
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
