#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Parser
{
private:
    std::string mtx_file_name;
    bool do_check;
    size_t mtx_dim;

    double t_min, t_max;
    double c_step;
    double d_min;
    double alpha;
    double t_step;

    bool h_vector_provided;
public:
    void parse_args(int _argc, char **_argv);

    Parser();

    bool use_rand_mtx();
    [[nodiscard]] size_t get_mtx_dim() const {return mtx_dim;};
    [[nodiscard]] bool check() const {return do_check;};
    [[nodiscard]] std::string get_mtx_file_name() const {return mtx_file_name; };

    [[nodiscard]] double get_t_min() {return t_min;};
    [[nodiscard]] double get_t_max() {return t_max;};
    [[nodiscard]] double get_c_step() {return c_step;};
    [[nodiscard]] double get_d_min() {return d_min;};
    [[nodiscard]] double get_alpha() {return alpha;};
    [[nodiscard]] double get_t_step() {return t_step;};
    [[nodiscard]] bool get_h_vector_provided() {return h_vector_provided;};
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cmd_parser.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
