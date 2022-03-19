#pragma once
#include <algorithm>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Parser::Parser()
{
    mtx_file_name = "";
    do_check = false;
    mtx_dim = 0;

    t_min = 1, t_max = 10;
    c_step = 1; // as in the paper
    d_min = 0.0001; // as in the paper
    alpha = 0.5;
    batch_file_name = "";
    num_gpus = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool Parser::use_rand_mtx()
{
    if(mtx_file_name.empty())
        return true;
    else
        return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Parser::parse_args(int _argc, char **_argv)
{
    // get params from cmd line
    for (int i = 1; i < _argc; i++)
    {
        std::string option(_argv[i]);

        if ((option == "-mtx") || (option == "-mtx-file") || (option == "-hmtx") || (option == "-mtx-file"))
        {
            mtx_file_name = std::string(_argv[++i]);
        }

        if ((option == "-check"))
        {
            do_check = true;
        }

        if ((option == "-dim") || (option == "-mtx-dim"))
        {
            mtx_dim = atoi(_argv[++i]);
        }

        if ((option == "-t_min"))
        {
            t_min = to_float(std::string(_argv[++i]));
        }

        if ((option == "-t_max"))
        {
            t_max = to_float(std::string(_argv[++i]));
        }

        if ((option == "-c_step"))
        {
            c_step = to_float(std::string(_argv[++i]));
        }

        if ((option == "-d_min"))
        {
            d_min = to_float(std::string(_argv[++i]));
        }

        if ((option == "-alpha") || (option == "-a"))
        {
            alpha = to_float(std::string(_argv[++i]));
        }

        if ((option == "-batch"))
        {
            batch_file_name = std::string(_argv[++i]);
            parse_batch_file();
        }

        if ((option == "-gpus") || (option == "-ngpu"))
        {
            num_gpus = std::atoi(_argv[++i]);
        }

        if ((option == "-help") || (option == "-h"))
        {
            std::cout << "-dim [N], specifies the size of input matrix (required argument)" << std::endl;
            std::cout << "-mtx [file_name], specifies the name of input file, which contains the matrix only (without h vector). If not specified, random matrix is generated. H-vector initialized with zeros. (optional argument)" << std::endl;
            std::cout << "-check, executes verification with sequential algorithm on CPU (turned off by default)" << std::endl;

            std::cout << "-t_min [N], specifies t_min (optional, default is 1)" << std::endl;
            std::cout << "-t_max [N], specifies t_max (optional, default is 10)" << std::endl;
            std::cout << "-c_step [N], specifies c_step (optional, default is 1)" << std::endl;
            std::cout << "-d_min [N], specifies d_min (optional, default is 0.0001)" << std::endl;
            std::cout << "-alpha [N], specifies alpha (optional, default is 0.5)" << std::endl;
            std::cout << "-batch [file_name], specifies the name of file, which contains information about multiple runs with different parameters {tmin, tmax, cstep, alpha}" << std::endl;
            std::cout << "-ngpu [N], set N as the maximum number of GPUs used. Only available for batched mode." << std::endl;
            std::cout << "-help, prints this message" << std::endl;

            throw "Help is requested, aborting...";
        }
    }

    if(mtx_dim == 0)
    {
        std::cout << "Error! matrix dimension is unset";
        throw "Aborting...";
    }

    if(!batch_file_provided())
    {
        std::cout << "t_min:" << t_min << std::endl;
        std::cout << "t_max:" << t_max << std::endl;
        std::cout << "c_step:" << c_step << std::endl;
        std::cout << "d_min:" << d_min << std::endl;
        std::cout << "alpha:" << alpha << std::endl;
    }
    else
    {
        std::cout << "batched mode:" << batch_file_provided() << std::endl;
    }
    if(batch_file_provided() && num_gpus > 0)
        std::cout << "ngpus:" << num_gpus << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Parser::parse_batch_file()
{
    batches_data.clear();

    std::ifstream file(batch_file_name);
    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        BatchInfo info;
        ss >> info.t_min;
        ss >> info.t_max;
        ss >> info.c_step;
        ss >> info.alpha;
        batches_data.push_back(info);
    }
    file.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Parser::sort_batches()
{
    if(batch_file_provided())
    {
        std::sort(batches_data.begin(), batches_data.end(),
             [](const BatchInfo & a, const BatchInfo & b) -> bool
             {
                 return a.work_amount() > b.work_amount();
             });
        for(auto i: batches_data)
            std::cout << i.work_amount() << std::endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
