#include <iostream>
#include <vector>
#include <random>
#include <math.h>

template <typename T>
class SquareMatrix
{
private:
    std::vector<T> data; // use one-dim vector for optimizations
    size_t dim_size;
public:
    explicit SquareMatrix(size_t _size): data(_size*_size, 0)
    {
        dim_size = _size;
    }

    inline T get(size_t i, size_t j) const
    {
        return data[i*dim_size + j];
    }

    inline void set(size_t i, size_t j, T _val)
    {
        data[i*dim_size + j] = _val;
    }

    void rand()
    {
        for(size_t j = 0; j < dim_size; j++)
        {
            for(size_t i = 0; i < dim_size; i++)
            {
                if(i > j)
                {
                    set(i, j, 0);
                }
            }
        }
    }
};

template <typename T>
int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

template <typename T>
void print(std::vector<T> &_data)
{
    for(auto &i: _data)
        std::cout << i << ' ';
    std::cout << std::endl;
}

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

template <typename T>
void read(std::vector<T> &_data)
{
    for(auto &i: _data)
        std::cout << i << ' ';
    std::cout << std::endl;
}

template <typename T>
auto seq_mars(SquareMatrix<T> &_J_mat,
              std::vector<T> &_h,
              size_t _n,
              int _t_min,
              int _t_max,
              T _c_step,
              T _d_min,
              T _alpha)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<T> uni(-1, 1);

    std::vector<T> s(_n, 0);
    std::vector<T> s_new(_n, 0);
    std::vector<T> phi(_n, 0);

    T current_temperature = 0, temperature = 0;
    for(int t = _t_min; t < _t_max; t++)
    {
        for(auto &s_i: s)
        {
            s_i = uni(rng);
            std::cout << s_i << ' ';
        }
        std::cout << std::endl;

        current_temperature = temperature; // t' = t

        while(current_temperature > 0)
        {
            T d = 0;
            current_temperature -= _c_step;
            do
            {
                for(size_t i = 0; i < phi.size(); i++) // is it 0 or 1?
                {
                    T sum = 0;
                    for(size_t j = 0; j < _n; j++)
                    {
                        sum = _J_mat.get(i, j) * s[j];
                    }
                    phi[i] = sum + _h[i];

                    if(current_temperature > 0)
                        s_new[i] = _alpha*(-tanh(phi[i] / current_temperature)) + (1 - _alpha)*s[i];
                    else
                        s_new[i] = -sign(phi[i]);

                    if(abs(s_new[i] - s[i]) > d)
                    {
                        d = abs(s_new[i] - s[i]);
                    }

                    s[i] = s_new[i];
                }
            } while(d < _d_min);
        }
    }

    return s;
}

#define base_type double

int main()
{
    size_t n = 10;
    int t_min = 0, t_max = 10;
    base_type c_step = 3;
    base_type d_min = 10;
    base_type alpha = 2;

    std::vector<base_type> h(n, 0);
    SquareMatrix<base_type> J(n);

    auto s = seq_mars(J, h, n, t_min, t_max, c_step, d_min, alpha);
    std::cout << "result: ";
    print(s);

    std::cout << "energy: " << dot_product(vxm(s, J), s) + dot_product(h, s) << std::endl;

    return 0;
}
