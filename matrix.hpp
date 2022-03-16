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

/*
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
}*/

size_t get_number_of_lines_in_file(const std::string &_file_name)
{
    char newLine = '.';
    size_t numLines = 0;
    std::string text;
    std::ifstream openFile(_file_name.c_str());

    std::cout << std::endl;

    if(!openFile)
    {
        return 0;
    }

    while(getline(openFile, text, '\n'))
    {
        for(unsigned int i=0; i< text.length(); i++)
        {
            if(text.at(i) == newLine)
            {
                numLines++;
            }
        }
    }

    return numLines;
}

bool check_if_h_vector_provided(const std::string &_file_name, size_t dim_size)
{
    size_t lines_number = get_number_of_lines_in_file(_file_name);
    if(lines_number == 0)
        return false;

    std::cout << "lines number: " << lines_number << std::endl;
    size_t expected_lines_number_with_h = (dim_size - 1)*((dim_size - 1) + 1)/2 + dim_size;
    size_t expected_lines_number_without_h = (dim_size - 1)*((dim_size - 1) + 1)/2;
    std::cout << "expected_lines_number_WITH_h: " << expected_lines_number_with_h << std::endl;
    std::cout << "expected_lines_number_WITHOUT_h: " << expected_lines_number_without_h << std::endl;

    bool h_is_provided;
    if(lines_number == expected_lines_number_with_h)
    {
        h_is_provided = true;
    }
    else if(lines_number == expected_lines_number_without_h)
    {
        h_is_provided = false;
    }
    else
    {
        std::cout << "input file " << _file_name << " does not have enough lines to construct square symmetric matrix" << std::endl;
        throw "Aborting...";
    }

    return h_is_provided;
}

template <typename T>
void SquareMatrix<T>::read_from_file(const std::string &_file_name)
{
    bool h_is_provided = check_if_h_vector_provided(_file_name, dim_size);

    std::ifstream file_desc;
    file_desc.open(_file_name);

    if(file_desc.is_open())
    {
        if(h_is_provided) // then skip h vector here, and read it later
        {
            for (size_t i = 0; i < dim_size; i++)
            {
                std::string line;
                if(file_desc.eof())
                {
                    std::cout << "Error! matrix dimension and number of elements in file mismatches" << std::endl;
                    throw "Aborting...";
                }
                file_desc >> line;
            }
        }

        for(size_t j = 0; j < dim_size; j++)
        {
            for (size_t i = 0; i < dim_size; i++)
            {
                if(i > j)
                {
                    std::string line;
                    if(file_desc.eof())
                    {
                        std::cout << "Error! matrix dimension and number of elements in file mismatches" << std::endl;
                        throw "Aborting...";
                    }
                    file_desc >> line;
                    T val = 0;
                    if(is_float(line))
                    {
                        val = to_float(line);
                    }

                    set(i, j, val);
                    set(j, i, val);
                }
                if(i == j)
                    set(i, j, 0);
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
