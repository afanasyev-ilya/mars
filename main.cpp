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
            J.read_from_file(parser.get_mtx_file_name());
            J.print();
        }

        const size_t n = J.get_dim_size();
        base_type t_min = parser.get_t_min(), t_max = parser.get_t_max();
        base_type c_step = parser.get_c_step();
        base_type d_min = parser.get_d_min();
        base_type alpha = parser.get_alpha();
        base_type t_step = parser.get_t_step();

        std::vector<base_type> h(n, 0);

        auto parallel_s = parallel_mars(J, h, n, t_min, t_max, c_step, d_min, alpha, t_step);
        std::cout << "parallel result: ";
        print(parallel_s);
        base_type parallel_energy = dot_product(vxm(parallel_s, J), parallel_s) + dot_product(h, parallel_s);
        std::cout << "parallel energy: " << parallel_energy << std::endl;

        if(parser.check())
        {
            auto sequential_s = sequential_mars(J, h, n, t_min, t_max, c_step, d_min, alpha, t_step);
            base_type sequential_energy = dot_product(vxm(sequential_s, J), sequential_s) + dot_product(h, sequential_s);
            std::cout << "sequential energy: " << sequential_energy << std::endl;

            if(parallel_energy == sequential_energy)
            {
                std::cout << "energies are correct!" << std::endl;
            }
            else
            {
                std::cout << "energies are NOT correct!" << std::endl;
            }

            if(parallel_s == sequential_s)
            {
                std::cout << "vectors are the same!" << std::endl;
            }
            else
            {
                std::cout << "vectors are NOT the same!" << std::endl;
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

