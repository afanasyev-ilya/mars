#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "settings.hpp"
#include "helpers.hpp"
#include "matrix.h"
#include "cmd_parser.h"
#include "parallel.hpp"
#include "sequential.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

