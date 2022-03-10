#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <fstream>
#include <sstream>
#include <iomanip>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool is_float(std::string my_string)
{
    std::istringstream iss(my_string);
    float f;
    iss >> std::noskipws >> f;
    return iss.eof() && !iss.fail();
}

double to_float(std::string my_string)
{
    std::istringstream iss(my_string);
    double f = 0;
    iss >> std::noskipws >> f;
    return f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class SquareMatrix
{
private:
    std::vector<T> data; // use one-dim vector for optimizations
    size_t dim_size;
public:
    explicit SquareMatrix(size_t _size = 1): data(_size*_size, 0)
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

    inline void set(size_t i, T _val)
    {
        data[i] = _val;
    }

    void fill_with_rands(size_t &_dim_size)
    {
        dim_size = _dim_size;
        data.resize(dim_size*dim_size);

        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<T> uni(-1, 1);

        for(size_t j = 0; j < dim_size; j++)
        {
            for(size_t i = 0; i < dim_size; i++)
            {
                if(i >= j) // to ensure it is symmetric
                {
                    T val = uni(rng);
                    set(i, j, val);
                    set(j, i, val);
                }
            }
        }
    }

    void print()
    {
        for(size_t j = 0; j < dim_size; j++)
        {
            for (size_t i = 0; i < dim_size; i++)
            {
                std::cout << get(i, j) << std::setprecision(4) << " ";
            }
            std::cout << std::endl;
        }
    }

    void read_from_file(const std::string &_file_name)
    {
        std::ifstream file_desc;
        file_desc.open(_file_name);

        std::vector<T> tmp_vals;

        if(file_desc.is_open())
        {
            while (!file_desc.eof())
            {
                std::string line;
                file_desc >> line;
                //std::cout << line << std::endl;
                if(is_float(line))
                {
                    tmp_vals.push_back(to_float(line));
                }
            }

            size_t num_elems = tmp_vals.size();

            dim_size = (sqrt(8*num_elems + 1) - 1)/2; // assuming we have just read N*(N+1)/2 elements, and try to find N
            std::cout << "dim size: " << dim_size << std::endl;
            std::cout << "elements read: " << num_elems << std::endl;
            size_t cnt = 0;
            data.resize(dim_size*dim_size);
            for(size_t j = 0; j < dim_size; j++)
            {
                for (size_t i = 0; i < dim_size; i++)
                {
                    if(i >= j)
                    {
                        set(i, j, tmp_vals[cnt]);
                        set(j, i, tmp_vals[cnt]);
                        cnt++;
                    }
                }
            }
        }
        else
        {
            throw "mtx file does not exist!";
        }

        file_desc.close();
    }

    [[nodiscard]] size_t get_dim_size() const {return dim_size;};
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void read_from_file(std::vector<T> &_vector, std::ifstream &_file_desc)
{

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
int sign(T val)
{
    return (T(0) < val) - (val < T(0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void print(std::vector<T> &_data)
{
    for(auto &i: _data)
        std::cout << i << ' ';
    std::cout << std::endl;
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

template <typename T>
void read(std::vector<T> &_data)
{
    for(auto &i: _data)
        std::cout << i << ' ';
    std::cout << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define base_type double

int main()
{
    try
    {
        SquareMatrix<base_type> J;
        if(false)
        {
            size_t dim_size = 10;
            J.fill_with_rands(dim_size);
        }
        else
        {
            J.read_from_file("test_mat.csv");
            J.print();
        }

        const size_t n = J.get_dim_size();
        int t_min = 0, t_max = 100;
        base_type c_step = 3;
        base_type d_min = 10;
        base_type alpha = 2;
        std::vector<base_type> h(n, 0);

        auto s = seq_mars(J, h, n, t_min, t_max, c_step, d_min, alpha);
        std::cout << "result: ";
        print(s);

        std::cout << "energy: " << dot_product(vxm(s, J), s) + dot_product(h, s) << std::endl;
    }
    catch (std::string error)
    {
        std::cout << error << std::endl;
    }
    catch (const char * error)
    {
        std::cout << error << std::endl;
    }


    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

