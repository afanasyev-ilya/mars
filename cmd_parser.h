#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Parser
{
private:
    std::string mtx_file_name;
    bool do_check;
    int mtx_dim;

    double t_min, t_max;
    double c_step;
    double d_min;
    double alpha;
    double t_step;
public:
    void parse_args(int _argc, char **_argv);

    Parser();

    bool use_rand_mtx();
    int get_mtx_dim() const {return mtx_dim;};
    bool check() const {return do_check;};
    std::string get_mtx_file_name() const {return mtx_file_name; };

    double get_t_min() {return t_min;};
    double get_t_max() {return t_max;};
    double get_c_step() {return c_step;};
    double get_d_min() {return d_min;};
    double get_alpha() {return alpha;};
    double get_t_step() {return t_step;};
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cmd_parser.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
