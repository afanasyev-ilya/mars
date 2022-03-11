compilation:

git clone https://github.com/afanasyev-ilya/mars.git
cd mars
make CXX=nvcc

For non-default CUDA Toolkit install location, please specify CUDA_DIR variable in the makefile.

running:

./bin/mars_cuda_cu -mtx test_mat.csv -check