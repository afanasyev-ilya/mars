/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define FULL_MASK 0xffffffff

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define VWARP_SIZE 32
#define VWARP_NUM 32
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double free_memory_size()
{
    size_t free, total;
    int id;
    cudaGetDevice( &id );
    cudaMemGetInfo( &free, &total );
    return free/1e9; // converting to GB
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
