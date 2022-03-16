compilation:

1) git clone https://github.com/afanasyev-ilya/mars.git

2) cd mars

3) make CXX=nvcc

For non-default CUDA Toolkit install location, please specify CUDA_DIR variable in the makefile.

run:

1) ./bin/mars_cuda_cu -dim 20 -mtx test_mat.csv 

<i>to run on matrix with(or without) h vector provided. In the case when h vector not provided, it is set to all zeroes.</i>

2) ./bin/mars_cuda_cu -dim 100

<i>to run on a larger random matrix (100x100 in this example)</i>

3) ./bin/mars_cuda_cu -help

<i>to get information about all available parameters</i>