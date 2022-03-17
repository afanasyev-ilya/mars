#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class SquareMatrix
{
private:
    std::vector<T> data; // use one-dim vector for optimizations
    int dim_size;
public:
    explicit SquareMatrix(int _size = 1): data(_size*_size, 0)
    {
        dim_size = _size;
    }

    inline T get(int i, int j) const
    {
        return data[i*dim_size + j];
    }

    inline void set(int i, int j, T _val)
    {
        data[i*dim_size + j] = _val;
    }

    inline void set(int i, T _val)
    {
        data[i] = _val;
    }

    void fill_with_rands(int &_dim_size);

    void print();

    void read_from_file(const std::string &_file_name);

    T* get_ptr() {return &data[0];};

    inline int get_dim_size() const {return dim_size;};
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "matrix.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
