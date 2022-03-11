#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_CUDA__
#include "cuda_mars.hpp"
#else
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
    std::cout << "CUDA implementation of mars is not activated during compilation, please set __USE_CUDA__ flag in settings" << std::endl;
    throw "Aborting...";
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
auto openmp_mars(SquareMatrix<T> &_J_mat,
                 std::vector<T> &_h,
                 size_t _n,
                 int _t_min,
                 int _t_max,
                 T _c_step,
                 T _d_min,
                 T _alpha,
                 T _t_step)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<T> uni(-1, 1);

    std::vector<T> s(_n, 0);
    std::vector<T> s_trial(_n, 0);
    std::vector<T> phi(_n, 0);

    T current_temperature = 0;
    for(base_type temperature = _t_min; temperature < _t_max; temperature += _t_step)
    {
        for(auto &s_i: s)
        {
            s_i = uni(rng);
        }

        current_temperature = temperature; // t' = t

        while(current_temperature > 0)
        {
            T d = 0;
            current_temperature -= _c_step;
            do
            {
                for(size_t i = 0; i < phi.size(); i++)
                {
                    T sum = 0;
                    for(size_t j = 0; j < _n; j++)
                    {
                        sum += _J_mat.get(i, j) * s[j];
                    }
                    phi[i] = sum + _h[i];

                    if(current_temperature > 0)
                    {
                        s_trial[i] = _alpha * (-tanh(phi[i] / current_temperature)) + (1 - _alpha) * s[i];
                    }
                    else
                    {
                        s_trial[i] = -sign(phi[i]);
                    }

                    if(abs(s_trial[i] - s[i]) > d)
                    {
                        d = abs(s_trial[i] - s[i]);
                    }

                    s[i] = s_trial[i];
                }
            } while(d < _d_min);
        }
    }

    return s;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
auto parallel_mars(SquareMatrix<T> &_J_mat,
                   std::vector<T> &_h,
                   size_t _n,
                   int _t_min,
                   int _t_max,
                   T _c_step,
                   T _d_min,
                   T _alpha,
                   T _t_step)
{
    #ifdef __USE_CUDA__
    return cuda_mars(_J_mat, _h, _n, _t_min, _t_max, _c_step, _d_min, _alpha, _t_step);
    #else
    return 0;
    #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
