#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Parser::Parser()
{
    mtx_file_name = "";
    do_check = false;
    mtx_dim = 0;

    t_min = 1, t_max = 10;
    c_step = 1; // as in the paper
    d_min = 0.0001; // as in the paper
    alpha = 0.3;
    t_step = 0.001;
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

        if ((option == "-t_step"))
        {
            t_step = to_float(std::string(_argv[++i]));
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
            std::cout << "-t_step [N], specifies step between t_min and t_max (optional, default is 0.1)" << std::endl;
            std::cout << "-help, prints this message" << std::endl;

            throw "Help is requested, aborting...";
        }
    }

    if(mtx_dim == 0)
    {
        std::cout << "Error! matrix dimension is unset";
        throw "Aborting...";
    }

    std::cout << "t_min:" << t_min << std::endl;
    std::cout << "t_max:" << t_max << std::endl;
    std::cout << "c_step:" << c_step << std::endl;
    std::cout << "d_min:" << d_min << std::endl;
    std::cout << "alpha:" << alpha << std::endl;
    std::cout << "t_step:" << t_step << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
