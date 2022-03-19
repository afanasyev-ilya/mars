#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <omp.h>

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
            #ifdef __DEBUG_INFO__
            J.print();
            #endif
        }
        else
        {
            J.read_from_file(parser.get_mtx_file_name());
            #ifdef __DEBUG_INFO__
            J.print();
            #endif
        }

        // common params
        const int n = J.get_dim_size();
        base_type d_min = parser.get_d_min();

        // read h-vector
        std::vector<base_type> h(n, 0);
        if(check_if_h_vector_provided(parser.get_mtx_file_name(), parser.get_mtx_dim()))
        {
            read_from_file(h, parser.get_mtx_file_name());
            #ifdef __DEBUG_INFO__
            print(h);
            #endif
        }

        // run main computations
        if(!parser.batch_file_provided())
        {
            base_type t_min = parser.get_t_min(), t_max = parser.get_t_max();
            base_type c_step = parser.get_c_step();
            base_type alpha = parser.get_alpha();

            double parallel_time = 0;
            base_type parallel_energy = parallel_mars(J, h, n, t_min, t_max, c_step, d_min, alpha, parallel_time);

            if(parser.check())
            {
                double seq_time = 0;
                base_type sequential_energy = sequential_mars(J, h, n, t_min, t_max, c_step, d_min, alpha, seq_time);
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
            #ifdef __USE_CUDA__
            int num_gpus_installed = 0;
            if(!parser.num_gpus_is_set())
            {
                 cudaGetDeviceCount(&num_gpus_installed);
                 std::cout << "automatically detected to run on " << num_gpus_installed << " GPUs" << std::endl;
            }
            else
            {
                num_gpus_installed = parser.get_num_gpus();
            }

            // sort batches for better load balancing depending on the amount of work (t_max - tmin)/c_step
            parser.sort_batches();

            double t1 = omp_get_wtime();
            #pragma omp parallel for num_threads(num_gpus_installed) schedule(dynamic)
            for(int batch_pos = 0; batch_pos < parser.get_num_batches(); batch_pos++)
            {
                int tid = omp_get_thread_num(); // max = num_gpus_installed
                int attached_gpu = tid;
                cudaSetDevice(attached_gpu); // select which GPU we use
                std::cout << "attaching to " << attached_gpu << " gpu" << std::endl;
                BatchInfo info = parser.get_batch_info(batch_pos);
                double parallel_time = 0;
                base_type parallel_energy = parallel_mars<base_type>(J, h, n, info.t_min, info.t_max, info.c_step, d_min, info.alpha, parallel_time);
                #pragma omp critical
                {
                    std::cout << "batch № " << batch_pos << std::endl;
                    info.print();
                    std::cout << "min energy: " << parallel_energy << std::endl;
                }
            }
            double t2 = omp_get_wtime();
            std::cout << "processing whole batch time: " << (t2 - t1) << " seconds" << std::endl;
            #else
            std::cout << "Running in batched mode does not make sense without at least GPU installed, "
                         "and program compiled without CUDA support!" << std::endl;

            return 0;
            #endif
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

