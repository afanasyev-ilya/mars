#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <fstream>
#include <sstream>
#include <iomanip>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "settings.hpp"
#include "helpers.hpp"
#include "matrix.h"
#include "cmd_parser.h"
#include "parallel.hpp"
#include "sequential.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);

        SquareMatrix<base_type> J;
        if(parser.use_rand_mtx())
        {
            std::cout << "Generating random matrix of size " << parser.get_mtx_dim() << std::endl;
            size_t dim_size = parser.get_mtx_dim();
            J.fill_with_rands(dim_size);
        }
        else
        {
            J.read_from_file(parser.get_mtx_file_name());
            J.print();
        }

        const size_t n = J.get_dim_size();
        base_type t_min = 1, t_max = 10;
        base_type c_step = 1; // as in the paper
        base_type d_min = 0.0001; // as in the paper
        base_type alpha = 0.4;
        base_type t_step = 1;
        std::vector<base_type> h(n, 0);

        auto parallel_s = parallel_mars(J, h, n, t_min, t_max, c_step, d_min, alpha, t_step);
        std::cout << "parallel result: ";
        print(parallel_s);
        std::cout << "parallel energy: " << dot_product(vxm(parallel_s, J), parallel_s) +
                dot_product(h, parallel_s) << std::endl;

        if(parser.check())
        {
            auto sequential_s = parallel_mars(J, h, n, t_min, t_max, c_step, d_min, alpha, t_step);
            if(parallel_s == sequential_s)
            {
                std::cout << "results are correct!" << std::endl;
            }
            else
            {
                std::cout << "results are NOT correct!" << std::endl;
                print(parallel_s);
                print(sequential_s);
            }
        }


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

