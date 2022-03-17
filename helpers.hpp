#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool is_float(std::string my_string)
{
    std::istringstream iss(my_string);
    float f;
    iss >> std::noskipws >> f;
    return iss.eof() && !iss.fail();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double to_float(std::string my_string)
{
    std::istringstream iss(my_string);
    double f = 0;
    iss >> std::noskipws >> f;
    return f;
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
    if(_data.size() < MAX_PRINTING_SIZE)
    {
        for(auto &i: _data)
            std::cout << i << ' ';
        std::cout << std::endl;
    }
    else
    {
        std::cout << "Warning! vector is too large to print! (" << _data.size() << ")" << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void read_from_file(std::vector<T> &_v, const std::string &_file_name)
{
    std::ifstream file_desc;
    file_desc.open(_file_name);

    if (file_desc.is_open())
    {
        for (int i = 0; i < _v.size(); i++)
        {
            std::string line;
            if (file_desc.eof())
            {
                std::cout << "Error! h-vector dimension and number of elements in file mismatches" << std::endl;
                throw "Aborting...";
            }
            file_desc >> line;
            if(is_float(line))
            {
                _v[i] = to_float(line);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
