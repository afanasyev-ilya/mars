#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <omp.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
T dot_product(const std::vector<T> &_v1, const std::vector<T> &_v2)
{
    if(_v1.size() != _v2.size())
        throw "Incorrect dims in dot product";
    T sum = 0;
    for(int i = 0; i < _v1.size(); i++)
        sum += _v1[i] * _v2[i];
    return sum;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
auto vxm(const std::vector<T> &_vector, const SquareMatrix<T> &_matrix)
{
    std::vector<T> result(_vector.size(), 0);
    for(int j = 0; j < result.size(); j++)
    {
        T sum = 0;
        for(int i = 0; i < result.size(); i++)
            sum += _vector[i]*_matrix.get(i, j);
        result[j] = sum;
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
auto sequential_mars(SquareMatrix<T> &_J_mat,
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
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<T> uni(-1, 1);

    std::vector<T> s(_n, 0);
    int iters = 0;

    T min_energy = std::numeric_limits<T>::max();
    double num_steps = (_t_max - _t_min)/_t_step;
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
                d = 0;
                for(int i = 0; i < _n; i++)
                {
                    T sum = 0;
                    for(int j = 0; j < _n; j++)
                    {
                        sum += _J_mat.get(i, j) * s[j];
                    }
                    T mean_field = sum + _h[i];

                    T s_trial_loc = 0;
                    if(current_temperature > 0)
                    {
                        s_trial_loc = _alpha * (-tanh(mean_field / current_temperature)) + (1 - _alpha) * s[i];
                    }
                    else if (mean_field >= 0)
                        s_trial_loc = -1;
                    else
                        s_trial_loc = 1;

                    if(abs(s_trial_loc - s[i]) > d)
                    {
                        d = abs(s_trial_loc - s[i]);
                    }

                    s[i] = s_trial_loc;
                }
            } while(d >= _d_min);
        }

        T energy = dot_product(vxm(s, _J_mat), s) + dot_product(_h, s);
        if(energy < min_energy)
            min_energy = energy;
    }

    double t2 = omp_get_wtime();
    std::cout << "CPU calculations finished in " << (t2 - t1) << " seconds" << std::endl;
    std::cout << "CPU min energy: " << std::setprecision(10) << min_energy << std::endl;

    _time = t2 - t1;

    return min_energy;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
