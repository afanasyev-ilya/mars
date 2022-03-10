#pragma once

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
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<T> uni(-1, 1);

    std::vector<T> s(_n, 0);
    std::vector<T> s_new(_n, 0);
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
                        sum = _J_mat.get(i, j) * s[j];
                    }
                    phi[i] = sum + _h[i];

                    if(current_temperature > 0)
                    {
                        s_new[i] = _alpha * (-tanh(phi[i] / current_temperature)) + (1 - _alpha) * s[i];
                    }
                    else
                    {
                        s_new[i] = -sign(phi[i]);
                    }

                    if(abs(s_new[i] - s[i]) > d)
                    {
                        d = abs(s_new[i] - s[i]);
                    }

                    s[i] = s_new[i];
                }

                std::cout << "par! d: " << d << " vs dmin: " << _d_min << std::endl;
            } while(d < _d_min);
        }
    }

    return s;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
