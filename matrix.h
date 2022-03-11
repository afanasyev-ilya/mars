#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class SquareMatrix
{
private:
    std::vector<T> data; // use one-dim vector for optimizations
    std::vector<int> unempty;
    size_t dim_size;
public:
    explicit SquareMatrix(size_t _size = 1): data(_size*_size, 0), unempty(_size*(_size+1), 0)
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

    void fill_with_rands(size_t &_dim_size);

    void print();

    void read_from_file(const std::string &_file_name);

    T* get_ptr() {return &data[0];};

    [[nodiscard]] inline size_t get_dim_size() const {return dim_size;};
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "matrix.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
