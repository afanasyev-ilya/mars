#pragma once
#include <tuple>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct BatchInfo
{
    double c_step;
    double t_min, t_max;
    double alpha;

    void print() const
    {
        std::cout << "tmin: " << t_min << " tmax: " << t_max << " c_step: " << c_step << " alpha: " << alpha << std::endl;
    }
};

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

    std::string batch_file_name;

    std::vector<BatchInfo> batches_data;

    void parse_batch_file();
public:
    void parse_args(int _argc, char **_argv);

    Parser();

    bool use_rand_mtx();
    int get_mtx_dim() const {return mtx_dim;};
    bool check() const {return do_check;};
    std::string get_mtx_file_name() const {return mtx_file_name; };
    std::string get_batch_file_name() const {return batch_file_name; };
    bool batch_file_provided() const {if(batch_file_name == "") return false; else return true;};

    int get_num_batches() {return (int)batches_data.size();};
    BatchInfo get_batch_info(int _batch_num) {return batches_data[_batch_num];};

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
