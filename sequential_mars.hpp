#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <omp.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
auto sequential_mars(SquareMatrix<T> &_J_mat,
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

    double t1 = omp_get_wtime();
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

    double t2 = omp_get_wtime();
    std::cout << "CPU calculations finished in " << (t2 - t1) << " seconds" << std::endl;

    return s;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
auto vxm(const std::vector<T> &_vector, const SquareMatrix<T> &_matrix)
{
    std::vector<T> result(_vector.size(), 0);
    for(size_t j = 0; j < result.size(); j++)
    {
        T sum = 0;
        for(size_t i = 0; i < result.size(); i++)
            sum += _vector[i]*_matrix.get(i, j);
        result[j] = sum;
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
T dot_product(const std::vector<T> &_v1, const std::vector<T> &_v2)
{
    if(_v1.size() != _v2.size())
        throw "Incorrect dims in dot product";
    T sum = 0;
    for(size_t i = 0; i < _v1.size(); i++)
        sum += _v1[i] * _v2[i];
    return sum;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////