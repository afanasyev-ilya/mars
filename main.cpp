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
#include "sequential_mars.hpp"
#include "parallel_mars.hpp"

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
            int dim_size = parser.get_mtx_dim();
            J.fill_with_rands(dim_size);
            J.print();
        }
        else
        {
            J.read_from_file(parser.get_mtx_file_name());
            J.print();
        }

        // common params
        const int n = J.get_dim_size();
        base_type t_step = parser.get_t_step();
        base_type d_min = parser.get_d_min();

        // read h-vector
        std::vector<base_type> h(n, 0);
        if(check_if_h_vector_provided(parser.get_mtx_file_name(), parser.get_mtx_dim()))
        {
            read_from_file(h, parser.get_mtx_file_name());
            print(h);
        }

        // run main computations
        if(!parser.batch_file_provided())
        {
            base_type t_min = parser.get_t_min(), t_max = parser.get_t_max();
            base_type c_step = parser.get_c_step();
            base_type alpha = parser.get_alpha();

            double parallel_time = 0;
            base_type parallel_energy = parallel_mars(J, h, n, t_min, t_max, c_step, d_min, alpha, t_step, parallel_time);

            if(parser.check())
            {
                double seq_time = 0;
                base_type sequential_energy = sequential_mars(J, h, n, t_min, t_max, c_step, d_min, alpha, t_step, seq_time);
                if(parallel_time > 0)
                    std::cout << "acceleration: " << seq_time / parallel_time << std::endl;

                if(fabs(parallel_energy - sequential_energy) < 0.00001) // numeric_limits::epslion does not fit here
                {
                    std::cout << "energies are correct!" << std::endl;
                }
                else
                {
                    std::cout << "energies are NOT correct!" << std::endl;
                }
            }
        }
        else
        {
            for(int batch_pos = 0; batch_pos < parser.get_num_batches(); batch_pos++)
            {
                BatchInfo info = parser.get_batch_info(batch_pos);
                double parallel_time = 0;
                base_type parallel_energy = parallel_mars<base_type>(J, h, n, info.t_min, info.t_max, info.c_step, d_min, info.alpha, t_step, parallel_time);
                std::cout << "batch " << batch_pos << " energy: " << parallel_energy << std::endl;
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

