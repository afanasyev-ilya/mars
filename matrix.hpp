#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SquareMatrix<T>::fill_with_rands(size_t &_dim_size)
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SquareMatrix<T>::print()
{
    if(dim_size*dim_size < MAX_PRINTING_SIZE)
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
    else
    {
        std::cout << "Warning! matrix is too large to print!" << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SquareMatrix<T>::read_from_file(const std::string &_file_name)
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
