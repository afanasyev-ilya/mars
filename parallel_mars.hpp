#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_CUDA__
#include "cuda_mars.hpp"
#else
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
    std::cout << "CUDA implementation of mars is not activated during compilation, please set __USE_CUDA__ flag in settings" << std::endl;
    throw "Aborting...";
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
auto parallel_mars(SquareMatrix<T> &_J_mat,
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
    #ifdef __USE_CUDA__
    return cuda_mars_warp_per_mean_field(_J_mat, _h, _n, _t_min, _t_max, _c_step, _d_min, _alpha, _t_step, _time);
    #else
    return sequential_mars(_J_mat, _h, _n, _t_min, _t_max, _c_step, _d_min, _alpha, _t_step, _time);
    #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
