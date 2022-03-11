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
#include "parallel_mars.hpp"
#include "sequential_mars.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);

        SquareMatrix<base_type> J(parser.get_mtx_dim());
        if(parser.use_rand_mtx())
        {
            std::cout << "Generating random matrix of size " << parser.get_mtx_dim() << std::endl;
            size_t dim_size = parser.get_mtx_dim();
            J.fill_with_rands(dim_size);
            J.print();
        }
        else
        {
            J.read_from_file(parser.get_mtx_file_name(), parser.get_h_vector_provided());
            J.print();
        }

        const size_t n = J.get_dim_size();
        base_type t_min = parser.get_t_min(), t_max = parser.get_t_max();
        base_type c_step = parser.get_c_step();
        base_type d_min = parser.get_d_min();
        base_type alpha = parser.get_alpha();
        base_type t_step = parser.get_t_step();

        std::vector<base_type> h(n, 0);
        if(parser.get_h_vector_provided())
        {
            read_from_file(h, parser.get_mtx_file_name());
            print(h);
        }

        auto parallel_energy = parallel_mars(J, h, n, t_min, t_max, c_step, d_min, alpha, t_step);

        if(parser.check())
        {
            auto sequential_energy = sequential_mars(J, h, n, t_min, t_max, c_step, d_min, alpha, t_step);

            if(parallel_energy == sequential_energy)
            {
                std::cout << "energies are correct!" << std::endl;
            }
            else
            {
                std::cout << "energies are NOT correct!" << std::endl;
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

