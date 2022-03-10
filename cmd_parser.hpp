#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Parser::Parser()
{
    mtx_file_name = "";
    do_check = false;
    mtx_dim = 10;
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

        if ((option == "-mtx") || (option == "-mtx-file"))
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
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
